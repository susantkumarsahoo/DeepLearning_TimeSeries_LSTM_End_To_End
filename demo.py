from src.pipelines.training_pipeline import main

if __name__ == "__main__":
    artifacts = main()
    print("\n" + "=" * 80)
    print("TRAINING PIPELINE EXECUTION COMPLETED")
    print("=" * 80)
    print("\nReturned Artifacts:")
    print(artifacts)

