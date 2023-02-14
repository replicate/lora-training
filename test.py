from predict_basic import Predictor

p = Predictor()
p.setup()
p.predict(
    instance_data="./quo.zip",
    task="face",
    seed=0,
    resolution=512,
)
