import subprocess
import json
prompt = f"""
Here are detected objects in an image:
{json.dumps(detections, indent=2)}
Describe the scene in natural language.
"""

result = subprocess.run(
    ["ollama", "run", "llama3"],
    input=prompt.encode(),
    stdout=subprocess.PIPE
)

print(result.stdout.decode())
