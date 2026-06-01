from pathlib import Path

from fastapi.testclient import TestClient

from api.server import app
from logger import logger

SOUND_FILE = (
    Path(__file__).parent
    / "data/sound_files/sound_files/2022-06-13--20-06-14_1__segment0.wav"
)

client = TestClient(app)

def test_spectral_inference_with_tabular():
    with SOUND_FILE.open("rb") as f:
        response = client.post(
            "/spectral-inference",
            files={"audio_file": ("test.wav", f, "audio/wav")},
            data={
                "hive_temp": "32.5",
                "hive_humidity": "50",
                "hive_pressure": "292",
                "frames": "10",
                "weather_temp": "20",
                "weather_humidity": "50",
                "weather_pressure": "1013",
                "wind_speed": "2.5",
                "cloud_coverage": "75",
                "date": "2022-06-08T14:52:28",
                "device": "1",
                "hive_number": "5",
            },
        )
    body = response.json()
    logger.info(f"API response: {body}, status code: {response.status_code}")


if __name__ == "__main__":
    test_spectral_inference_with_tabular()
