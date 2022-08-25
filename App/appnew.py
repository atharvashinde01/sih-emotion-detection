import uvicorn
from fastapi import FastAPI
from BaseModel import Base
# from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

import joblib 
pipe_lr = joblib.load(open("models/emotion_classifier_pipe_lr_03_june_2021.pkl","rb"))


@app.post("/prediction")
def predict_emotions(data:Base):
	# results = pipe_lr.predict([docx])
	data=data.dict()
	Emotion=data["Emotion"]
	Text=data["Text"]
	Clean_Text=data["Clean_Text"]
	results = pipe_lr.predict([Emotion, Text, Clean_Text])

	return results[0]


@app.get("/probablity")
def get_prediction_proba(docx):
	emotions_emoji_dict = {"anger":"😠","disgust":"🤮", "fear":"😨😱", "happy":"🤗", "joy":"😂", "neutral":"😐", "sad":"😔", "sadness":"😔", "shame":"😳", "surprise":"😮"}

	results = pipe_lr.predict_proba([docx])
	return results

if __name__ == "__main__":
    # app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
    uvicorn.run(app, host="127.0.0.1" ,port=8000)