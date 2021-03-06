build:
	@echo "Building ecosystem for OfflineAssistant"
	docker build TTS/german/male -t mtts:de
	docker build STT/german -t mstt:de

run:
	docker run -d -p 5002:5002 mtts:de
	docker run -d -p 5003:5003 mstt:de
	docker run -d -p 5004:5002 synesthesiam/mozillatts
	docker run -d -p 5005:5003 mstt:en