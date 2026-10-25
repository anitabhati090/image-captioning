from captioning.pipeline import CaptioningPipeline


class FakeModel:
    def caption(self, image, **kwargs):
        return "arafed a picture of a dog on the grass"


def test_pipeline_runs_without_real_model():
    result = CaptioningPipeline(FakeModel()).run(image=None, style="alt_text", language_code="en")
    assert result.styled_caption == "A dog on the grass."
    assert result.translated_caption == result.styled_caption
    assert result.translation_error is None
