import mlrun

if __name__ == "__main__":
    mlrun.set_environment("http://localhost:8080", artifact_path="./")
    project = mlrun.get_or_create_project("start", "./", user_project=True)
    data_gen_fn = project.set_function(
    "data_prep.py",
    name="data_prep",
    kind="job",
    image="mlrun/mlrun",
    handler="breast_cancer_generator",
    )
    project.save()  # save the project with the latest config

    gen_data_run = project.run_function("data_prep", local=True)
    print(f"Generate data: {gen_data_run.state()}")
    print(f"Output: {gen_data_run.outputs}")
    print(gen_data_run.artifact("dataset").as_df().head())

    # Import the function
    trainer = mlrun.import_function("hub://auto_trainer")

    trainer_run = project.run_function(
        trainer,
        inputs={"dataset": gen_data_run.outputs["dataset"]},
        params={
            "model_class": "sklearn.ensemble.RandomForestClassifier",
            "train_test_split_size": 0.2,
            "label_columns": "label",
            "model_name": "cancer",
        },
        handler="train",
    )