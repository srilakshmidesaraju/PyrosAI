"""Optional CNN land-cover verification for high-risk anomalies.

For events that are both anomalous and high risk, fetch recent imagery of the
coordinate and ask a CNN what kind of ground it is. "Confirms industrial area"
next to an FRP spike is corroboration a thermal sensor cannot give on its own.

Everything here is best-effort. The pipeline has already produced a complete
answer before this stage runs, so any failure - no credentials, no network, no
model weights - degrades to a weaker result and never raises.

Degradation ladder, in order:
  1. Sentinel-2 quicklook via Copernicus Data Space   (needs SENTINEL_* in .env)
  2. Esri World Imagery tile for the same coordinate  (keyless, always available)
  3. no image at all -> cnn_status="no_image", pipeline continues

Model ladder:
  1. models/efficientnet_eurosat.pth  -> real 10-class EuroSAT land cover
  2. ImageNet-pretrained EfficientNet-B0 -> a weak hint, clearly labelled as
     such. It is NOT reported as a land-cover class: a randomly initialised
     10-class head would emit a confident-looking label with no basis, which is
     worse than saying nothing.
"""

from __future__ import annotations

import io
import math
import os
import sys

import requests
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT = os.path.join(ROOT, "models", "efficientnet_eurosat.pth")

EUROSAT_CLASSES = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
                   "Industrial", "Pasture", "PermanentCrop", "Residential",
                   "River", "SeaLake"]

CONTEXT = {
    "Industrial": "Confirms industrial area — supports an industrial fire or flare",
    "Residential": "Residential area — a fire here is a public-safety concern",
    "Forest": "Confirms forested terrain — wildfire likely",
    "AnnualCrop": "Confirms cropland — agricultural burn likely",
    "PermanentCrop": "Confirms orchard or plantation — agricultural burn likely",
    "HerbaceousVegetation": "Scrub or grassland — vegetation fire likely",
    "Pasture": "Pasture — grassland burn likely",
    "Highway": "Transport corridor — verify against roadside burning",
    "River": "Water body — detection may be a sun-glint false positive",
    "SeaLake": "Water body — detection may be a sun-glint false positive",
}

CDSE_TOKEN_URL = ("https://identity.dataspace.copernicus.eu/auth/realms/CDSE"
                  "/protocol/openid-connect/token")
CDSE_SEARCH = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
ESRI_TILE = ("https://server.arcgisonline.com/ArcGIS/rest/services"
             "/World_Imagery/MapServer/tile/{z}/{y}/{x}")

MIN_RISK = 60.0
ESRI_ZOOM = 15


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------

_MODEL = None
_MODE = None


def load_model():
    """Return (model, mode). mode is 'eurosat' or 'imagenet' or None."""
    global _MODEL, _MODE
    if _MODEL is not None or _MODE == "unavailable":
        return _MODEL, _MODE

    try:
        import torch
        from torchvision import models
    except ImportError:
        print("  torch/torchvision not installed — vision stage skipped")
        _MODE = "unavailable"
        return None, _MODE

    try:
        if os.path.exists(CHECKPOINT):
            model = models.efficientnet_b0(weights=None)
            in_f = model.classifier[1].in_features
            model.classifier[1] = torch.nn.Linear(in_f, len(EUROSAT_CLASSES))
            model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"))
            _MODE = "eurosat"
            print(f"  model: EfficientNet-B0 fine-tuned on EuroSAT ({CHECKPOINT})")
        else:
            model = models.efficientnet_b0(weights="IMAGENET1K_V1")
            _MODE = "imagenet"
            print("  WARNING: models/efficientnet_eurosat.pth not found.")
            print("           Falling back to ImageNet weights. Imagery is still")
            print("           fetched and attached, but no land-cover class is")
            print("           claimed — an untrained 10-class head would invent one.")
        model.eval()
        _MODEL = model
    except Exception as err:                          # noqa: BLE001 - never fatal
        print(f"  model load failed ({type(err).__name__}: {err}) — vision skipped")
        _MODE = "unavailable"
        return None, _MODE

    return _MODEL, _MODE


def classify_image(img) -> tuple[str | None, float]:
    """(class name, confidence). Returns (None, 0.0) unless a real EuroSAT head."""
    model, mode = load_model()
    if model is None or mode != "eurosat":
        return None, 0.0

    import torch
    from torchvision import transforms

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    with torch.no_grad():                              # CPU inference, no autograd
        logits = model(tf(img.convert("RGB")).unsqueeze(0))
        probs = torch.softmax(logits, dim=1)[0]
        idx = int(probs.argmax())
    return EUROSAT_CLASSES[idx], round(float(probs[idx]), 3)


# --------------------------------------------------------------------------
# imagery
# --------------------------------------------------------------------------

def cdse_token() -> str | None:
    user, password = os.getenv("SENTINEL_USER", ""), os.getenv("SENTINEL_PASS", "")
    if not user or not password or user.startswith("your_"):
        return None
    try:
        resp = requests.post(CDSE_TOKEN_URL, timeout=30, data={
            "client_id": "cdse-public", "grant_type": "password",
            "username": user, "password": password,
        })
        if resp.status_code == 200:
            return resp.json().get("access_token")
        print(f"  Copernicus auth failed (HTTP {resp.status_code}) — using Esri imagery")
    except requests.RequestException as err:
        print(f"  Copernicus auth error ({type(err).__name__}) — using Esri imagery")
    return None


def sentinel_quicklook(lat: float, lon: float, token: str):
    """Most recent low-cloud Sentinel-2 quicklook near a coordinate."""
    d = 0.02
    flt = (f"Collection/Name eq 'SENTINEL-2' and "
           f"OData.CSC.Intersects(area=geography'SRID=4326;POLYGON(("
           f"{lon-d} {lat-d},{lon+d} {lat-d},{lon+d} {lat+d},"
           f"{lon-d} {lat+d},{lon-d} {lat-d}))') and "
           f"Attributes/OData.CSC.DoubleAttribute/any(a:a/Name eq 'cloudCover' and "
           f"a/OData.CSC.DoubleAttribute/Value lt 20.0)")
    try:
        resp = requests.get(CDSE_SEARCH, timeout=45, headers={
            "Authorization": f"Bearer {token}"},
            params={"$filter": flt, "$orderby": "ContentDate/Start desc", "$top": 1})
        if resp.status_code != 200 or not resp.json().get("value"):
            return None
        pid = resp.json()["value"][0]["Id"]
        ql = requests.get(
            f"https://catalogue.dataspace.copernicus.eu/odata/v1/Assets({pid})/$value",
            headers={"Authorization": f"Bearer {token}"}, timeout=60)
        if ql.status_code == 200:
            from PIL import Image
            return Image.open(io.BytesIO(ql.content))
    except Exception:                                  # noqa: BLE001
        return None
    return None


def esri_tile(lat: float, lon: float, zoom: int = ESRI_ZOOM):
    """Keyless Esri World Imagery tile covering the coordinate."""
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    url = ESRI_TILE.format(z=zoom, x=x, y=y)
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            from PIL import Image
            return Image.open(io.BytesIO(resp.content)), url
    except Exception:                                  # noqa: BLE001
        pass
    return None, None


# --------------------------------------------------------------------------
# stage
# --------------------------------------------------------------------------

def verify_event(event: dict, token: str | None) -> None:
    lat, lon = event["latitude"], event["longitude"]
    event.update(cnn_status="no_image", cnn_result=None,
                 cnn_confidence=None, cnn_context=None, cnn_image=None)

    img, source, url = None, None, None
    if token:
        img = sentinel_quicklook(lat, lon, token)
        source = "sentinel2" if img else None
    if img is None:
        img, url = esri_tile(lat, lon)
        source = "esri_fallback" if img else None
    if img is None:
        return

    event["cnn_image"] = url
    label, conf = classify_image(img)

    if label:
        event.update(cnn_status=source, cnn_result=label, cnn_confidence=conf,
                     cnn_context=CONTEXT.get(label, "Land cover identified"))
    else:
        _, mode = load_model()
        event.update(
            cnn_status=f"{source}_no_classifier",
            cnn_context=("Imagery retrieved for visual reference. No land-cover "
                         "class claimed: models/efficientnet_eurosat.pth is not "
                         "present, so the classifier head is untrained."
                         if mode == "imagenet" else
                         "Imagery retrieved for visual reference; classifier unavailable."))


def run(events: list[dict], min_risk: float = MIN_RISK,
        verbose: bool = True) -> list[dict]:
    targets = [e for e in events
               if e.get("anomaly_flag") and e.get("risk_score", 0) >= min_risk]

    if not targets:
        if verbose:
            print(f"  no events match anomaly_flag AND risk >= {min_risk:.0f} — nothing to verify")
        return events

    if verbose:
        print(f"  {len(targets)} high-risk anomalies queued for imagery verification")

    model, mode = load_model()
    if mode == "unavailable":
        for e in targets:
            e.update(cnn_status="unavailable", cnn_result=None,
                     cnn_confidence=None, cnn_context=None, cnn_image=None)
        return events

    token = cdse_token()
    if verbose and not token:
        print("  Copernicus credentials absent or rejected — using Esri imagery")

    for e in targets:
        try:
            verify_event(e, token)
        except Exception as err:                       # noqa: BLE001 - never fatal
            print(f"    {e['event_id']}: {type(err).__name__} — skipped")
            e.update(cnn_status="no_image", cnn_result=None,
                     cnn_confidence=None, cnn_context=None, cnn_image=None)

    if verbose:
        got = sum(1 for e in targets if e.get("cnn_image") or e.get("cnn_result"))
        named = sum(1 for e in targets if e.get("cnn_result"))
        print(f"  imagery retrieved for {got}/{len(targets)}; "
              f"land cover classified for {named}")
    return events


if __name__ == "__main__":
    from pipeline.anomaly import run as anomaly_run
    from pipeline.classify import run as classify_run
    from pipeline.cluster import run as cluster_run
    from pipeline.export import compute_risk
    from pipeline.features import run as features_run

    evs = anomaly_run(classify_run(features_run(cluster_run(verbose=False))), verbose=False)
    for ev in evs:
        ev["risk_score"] = compute_risk(ev)
    run(evs)
