# Quick Fix: Google Maps Not Showing

## The Problem
You're seeing "Google Maps API Key Required" instead of the map.

## Quick Solution (3 Steps)

### Step 1: Get Your API Key
1. Visit: https://console.cloud.google.com/
2. Create/Select project
3. Enable: **Maps JavaScript API**, **Geocoding API**, **Places API**
4. Create API Key in Credentials section

### Step 2: Set Environment Variable (PowerShell)
```powershell
$env:GOOGLE_MAPS_API_KEY = "YOUR_API_KEY_HERE"
```

### Step 3: Restart Backend
```powershell
# Stop current server (Ctrl+C)
# Then restart:
python backend/app.py
```

## Verify It Works

1. Open browser console (F12)
2. Visit: http://127.0.0.1:5000/api/config
3. Check if `mapsApiKey` has a value (not empty)

## Test Maps

1. Go to: http://127.0.0.1:5000/
2. Search for "Lucknow"
3. Check result page - map should appear
4. Check dashboard - map with markers should appear

## If Still Not Working

Check browser console (F12) for errors. Common issues:
- API key not set correctly
- API key restrictions blocking localhost
- Required APIs not enabled
- Billing not enabled (free tier should work)

## Note
The map will only show AFTER you:
1. Set the API key
2. Restart the backend
3. Refresh the browser page

