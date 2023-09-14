from roboflow import Roboflow
import os
import time

# load model
rf = Roboflow(api_key="3cR60WzeoK9LNrEVOyPT")
project = rf.workspace().project("mdp-project")
model = project.version(1).model

for image in os.listdir("test_images"):
    if image.endswith(".jpg"):
        start = time.time()
        print(model.predict(f"test_images/{image}", confidence=40, overlap=30).json())
        print(f"Time taken: {time.time() - start}")
        #visualize
        model.predict(f"test_images/{image}", confidence=40, overlap=30).save(f"result_images/{image}")


