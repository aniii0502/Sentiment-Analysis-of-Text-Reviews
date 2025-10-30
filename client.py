import requests

API_URL = "http://127.0.0.1:5000/predict"

test_reviews = [
    {
        "text": "This movie was absolutely fantastic! The acting was superb and the plot was thrilling."
    },
    {
        "text": "What a total waste of time. The plot was boring and the acting was dreadful. I would not recommend this."
    },
    {
        "text": "It was an okay movie. Not great, but not terrible either. Just average."
    },
    {
        "text": ""
    },
    {
        "some_other_key": "This should fail"
    }
]

print("--- Testing Sentiment Analysis API ---")

for i, review in enumerate(test_reviews):
    print(f"\nTest Case {i+1}: Sending -> {review}")
    try:
        response = requests.post(API_URL, json=review)
        
        if response.status_code == 200:
            print(f"✅ Success (200 OK): {response.json()}")
        else:
            print(f"❌ Error ({response.status_code}): {response.json()}")
            
    except requests.exceptions.ConnectionError:
        print("❌ FAILED: ConnectionError. Is the Flask server (app.py) running?")
        break
    except Exception as e:
        print(f"❌ FAILED: An unexpected error occurred: {e}")