# Compute datasets: verified 2026-10-22

All endpoints below returned HTTP 200 with no API key (CelesTrak rate-limits repeat pulls to once every 2 hours, so cache locally).

## Tier 1: core

| Dataset | What | Endpoint | Size / notes | License |
|---|---|---|---|---|
| **Epoch AI Frontier Data Centers** | Largest AI datacenters: H100-equivalents, power (MW), capex, owner, users, chip types, country, address | `https://epoch.ai/data/data_centers/data_centers.csv` | 95 sites, rich per-site fields | CC BY 4.0 |
| **Vast.ai public GPU pricing** | Per GPU model: cheapest/p10/median $/hr, offer counts, VRAM, ~90 days of daily median history, updated hourly | `https://storage.googleapis.com/vast-public-gpu-pricing/gpu-pricing-public.json` | 78 GPU models | Display/cite with attribution; bulk redistribution needs license |
| **OpenRouter models** | Every hosted LLM with pricing; filter `pricing.prompt == "0"` for free models | `https://openrouter.ai/api/v1/models` | 468 models, 20 free today | Public API |
| **ATLAS Global Data Center Map** | 18,110 facilities, 116 countries, 4,181 operators | GitHub `ringmast4r/Global-Data-Center-Map`: `datacenters.geojson` / `.csv` / `.parquet` | Only ~1/3 have city-level coordinates; the rest use region centroids | Attribution |

## Tier 2: network + physical layer

| Dataset | What | Endpoint | Notes |
|---|---|---|---|
| **TeleGeography Submarine Cables** | Undersea internet cables + landing stations | `https://www.submarinecablemap.com/api/v3/cable/cable-geo.json`, `.../landing-point/landing-point-geo.json` | 733 cables, 1,928 landing points |
| **PeeringDB** | Colocation facilities, internet exchanges, which networks are present where | `https://www.peeringdb.com/api/fac`, `/ix`, `/net` | Free; anonymous rate limits |
| **CelesTrak Starlink** | Orbit data for the whole Starlink fleet (space layer) | `https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=json` | Updates every 2 hours; repeat pulls get 403 until new data arrives |
| **NOAA SWPC** | Space weather (radiation/storms) for the orbital-compute angle | `https://services.swpc.noaa.gov/products/noaa-scales.json` | Free JSON |

## Tier 3: optional enrichment
- **EIA API**: hourly US grid demand by region (free key)
- **Electricity Maps**: live grid carbon intensity (free tier, key)
- **datacenter-atlas** (GitHub `kiankyars`): datacenter construction sites with cited sources, static JSON API, CC BY 4.0 (not yet verified)
- **Data Center Index** (`datacenterindex.ai/data`): ~900 campuses incl. planned, CSV/JSON, CC BY 4.0 (row counts inconsistent on its own page)

## Space / ocean compute
No public dataset of orbital or undersea compute exists. Starcloud-1 (H100, launched Nov 2025) and Google Project Suncatcher (prototypes planned with Planet by early 2027) have no live data feeds. To cover them, model them from CelesTrak + SWPC + sunlight/eclipse calculations.
