# Google Maps API Setup Guide

## Issue: Maps Not Showing

If you see "Google Maps would go here" or maps are not displaying, follow these steps:

## Step 1: Get Google Maps API Key

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable these APIs:
   - **Maps JavaScript API**
   - **Geocoding API**
   - **Places API** (for autocomplete)
4. Go to "Credentials" → "Create Credentials" → "API Key"
5. Copy your API key

## Step 2: Set Environment Variable

### Windows PowerShell:
```powershell
$env:GOOGLE_MAPS_API_KEY = "YOUR_API_KEY_HERE"
```

### Windows CMD:
```cmd
set GOOGLE_MAPS_API_KEY=YOUR_API_KEY_HERE
```

### Linux/Mac:
```bash
export GOOGLE_MAPS_API_KEY="YOUR_API_KEY_HERE"
```

## Step 3: Restart Backend Server

After setting the environment variable, restart your Flask backend:
```bash
python backend/app.py
```

## Step 4: Verify API Key is Loaded

1. Open browser console (F12)
2. Visit: `http://localhost:5000/api/config`
3. Check if `mapsApiKey` is returned (not empty)

## Step 5: Test Maps

1. Go to home page
2. Search for a location (e.g., "Lucknow")
3. Check result page - map should display
4. Check dashboard - map with markers and heatmap should display

## Troubleshooting

### Map shows "Loading Map..." forever
- Check browser console for errors
- Verify API key is set correctly
- Check if API key has proper permissions

### "Google Maps API Key Required" message
- API key not set in environment variable
- Backend not reading environment variable
- Restart backend after setting variable

### Heatmap not showing
- Make sure you have searched at least 2-3 locations
- Check browser console for errors
- Verify visualization library is loaded (should be automatic)

### Maps show but no markers
- Make sure you have search history
- Check if locations are valid UP locations
- Check browser console for geocoding errors

## API Key Restrictions (Recommended)

For security, restrict your API key:
1. Go to Google Cloud Console → Credentials
2. Click on your API key
3. Under "Application restrictions":
   - Select "HTTP referrers"
   - Add: `http://localhost:5000/*`
   - Add: `http://127.0.0.1:5000/*`
4. Under "API restrictions":
   - Select "Restrict key"
   - Choose: Maps JavaScript API, Geocoding API, Places API

## Cost Note

Google Maps API has a free tier:
- $200 free credit per month
- Usually enough for development/testing
- Monitor usage in Google Cloud Console

