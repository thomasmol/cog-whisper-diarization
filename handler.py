import runpod
from predict import Predictor

predictor = Predictor()
predictor.setup()

def handler(job):
    job_input = job["input"]
    output = predictor.predict(
        file_string=job_input.get("file_string"),
        file_url=job_input.get("file_url"),
        file=job_input.get("file"),
        num_speakers=job_input.get("num_speakers"),
        translate=job_input.get("translate", False),
        language=job_input.get("language"),
        prompt=job_input.get("prompt"),
    )
    return {
        "segments": output.segments,
        "language": output.language,
        "num_speakers": output.num_speakers,
        "speakers": output.speakers or {},
    }

runpod.serverless.start({"handler": handler})
