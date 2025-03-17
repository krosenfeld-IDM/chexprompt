import os

import openai
import chexprompt

engine = "gpt-4-1106-preview"  # Azure OpenAI deployment name


def test_evaluate_false_positive():
    evaluator = chexprompt.ReportEvaluator(engine=engine)

    reference_report = "The heart has normal size. The lungs are clear. There is no pleural effusion or pneumothorax. There is no focal airspace consolidation. There are no acute bony findings."
    candidate_report = "There is severe cardiomegaly. The lungs are clear. There is no pleural effusion or pneumothorax. There is no focal airspace consolidation. There are no acute bony findings."

    results = evaluator.evaluate(reference_report, candidate_report)

    expected_result = [
        {
            "clinically_insignificant": {
                "false_positive_comparison": 0,
                "false_positive_finding": 0,
                "incorrect_location": 0,
                "incorrect_severity": 0,
                "omission_comparison": 0,
                "omission_finding": 0,
            },
            "clinically_significant": {
                "false_positive_comparison": 0,
                "false_positive_finding": 1,
                "incorrect_location": 0,
                "incorrect_severity": 0,
                "omission_comparison": 0,
                "omission_finding": 0,
            },
        }
    ]

    assert results == expected_result


def test_evaluate_no_error():
    evaluator = chexprompt.ReportEvaluator(engine=engine)

    reference_report = "There is pleural effusion."
    candidate_report = "Evidence of pleural effusion."

    results = evaluator.evaluate(reference_report, candidate_report)

    expected_result = [
        {
            "clinically_insignificant": {
                "false_positive_comparison": 0,
                "false_positive_finding": 0,
                "incorrect_location": 0,
                "incorrect_severity": 0,
                "omission_comparison": 0,
                "omission_finding": 0,
            },
            "clinically_significant": {
                "false_positive_comparison": 0,
                "false_positive_finding": 0,
                "incorrect_location": 0,
                "incorrect_severity": 0,
                "omission_comparison": 0,
                "omission_finding": 0,
            },
        }
    ]

    assert results == expected_result
