import sys
from inference import predict_video


if len(sys.argv) != 2:
    print("Usage:")
    print("  python test.py path_to_video.avi")
    sys.exit(1)

video_path = sys.argv[1]

result = predict_video(video_path)

print("\n" + "=" * 55)
print("HUMAN ACTIVITY DETECTION")
print("=" * 55)

print(f"Predicted Activity : {result['activity']}")
print(f"Confidence         : {result['confidence'] * 100:.2f}%")

print("\nClass probabilities:")

for activity, probability in result["probabilities"].items():
    print(f"{activity:<15}: {probability * 100:.2f}%")

print("=" * 55)
