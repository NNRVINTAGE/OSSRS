import sys
import traceback

def run_all_tests():
    print("=== RUNNING PREPROCESSING TESTS ===")
    from tests.test_preprocessing import (
        test_synthetic_image_preprocessing,
        test_real_dataset_image_preprocessing
    )
    try:
        test_synthetic_image_preprocessing()
        print("  [PASS] test_synthetic_image_preprocessing")
        test_real_dataset_image_preprocessing()
        print("  [PASS] test_real_dataset_image_preprocessing")
    except Exception as e:
        print(f"  [FAIL] Preprocessing: {e}")
        traceback.print_exc()
        return False

    print("\n=== RUNNING FEATURE EXTRACTION TESTS ===")
    from tests.test_features import (
        test_feature_dimensions,
        test_feature_extraction_pipeline
    )
    try:
        test_feature_dimensions()
        print("  [PASS] test_feature_dimensions")
        test_feature_extraction_pipeline()
        print("  [PASS] test_feature_extraction_pipeline")
    except Exception as e:
        print(f"  [FAIL] Features: {e}")
        traceback.print_exc()
        return False

    print("\n=== RUNNING MATCHING & SIMILARITY TESTS ===")
    from tests.test_matching import (
        test_cosine_similarity,
        test_pairwise_feature_construction
    )
    try:
        test_cosine_similarity()
        print("  [PASS] test_cosine_similarity")
        test_pairwise_feature_construction()
        print("  [PASS] test_pairwise_feature_construction")
    except Exception as e:
        print(f"  [FAIL] Matching: {e}")
        traceback.print_exc()
        return False

    print("\nALL MODULE TESTS PASSED SUCCESSFULLY!")
    return True

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
