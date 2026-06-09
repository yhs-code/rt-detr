from ultralytics import RTDETR

if __name__ == "__main__":
    model = RTDETR("my_cfg/rtdetr-r18-SODGA-P2Detail.yaml")
    model.train(
        data="./dataset/A_drowning_person.yaml",
        cache=True,
        imgsz=640,
        epochs=100,
        batch=4,
        workers=4,
        device="0",
        optimizer="AdamW",
        deterministic=False,
        save=True,
        val=True,
        plots=True,
        project="runs/train",
        name="RT-DETR-R18-SODGA-P2Detail",
    )

