from pathlib import Path

from datasets import Dataset

from scripts.html_game_dataset import HTML_GAME_SYSTEM_PROMPT, HtmlGameDataset
from verl.utils.import_utils import load_extern_object


def _mock_html_game_dataset() -> HtmlGameDataset:
    dataset = HtmlGameDataset.__new__(HtmlGameDataset)
    dataset.image_key = "images"
    dataset.video_key = "videos"
    dataset.audio_key = "audios"
    dataset.processor = None
    return dataset


def test_build_messages_adds_html_game_system_prompt():
    dataset = _mock_html_game_dataset()

    messages = dataset._build_messages({"prompt": "Build a maze game."}, key="prompt")

    assert messages == [
        {"role": "system", "content": HTML_GAME_SYSTEM_PROMPT},
        {"role": "user", "content": "Build a maze game."},
    ]


def test_build_messages_preserves_existing_system_prompt():
    dataset = _mock_html_game_dataset()
    prompt = [
        {"role": "system", "content": "Existing policy."},
        {"role": "user", "content": "Build a maze game."},
    ]

    messages = dataset._build_messages({"prompt": prompt}, key="prompt")

    assert messages == prompt


def test_file_loaded_dataset_builds_messages_in_multiprocess_filter():
    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert messages[0] == {"role": "system", "content": HTML_GAME_SYSTEM_PROMPT}
            return [1, 2, 3]

    adapter_path = Path(__file__).parents[3] / "scripts" / "html_game_dataset.py"
    dataset_cls = load_extern_object(str(adapter_path), "HtmlGameDataset")
    dataset = dataset_cls.__new__(dataset_cls)
    dataset.image_key = "images"
    dataset.video_key = "videos"
    dataset.audio_key = "audios"
    dataset.processor = None
    dataset.tokenizer = FakeTokenizer()
    dataset.prompt_key = "prompt"
    dataset.filter_overlong_prompts = True
    dataset.apply_chat_template_kwargs = {}
    dataset.tool_schemas = None
    dataset.max_prompt_length = 10
    dataset.num_workers = 1
    dataset.serialize_dataset = True

    dataframe = Dataset.from_list([{"prompt": "Build a maze game."}])

    filtered = dataset.maybe_filter_out_long_prompts(dataframe)

    assert len(filtered) == 1


def test_file_loaded_dataset_filters_text_prompt_with_multimodal_processor():
    class FakeProcessorTokenizer:
        def __call__(self, text, **kwargs):
            assert text == "formatted prompt"
            return {"input_ids": [1, 2, 3]}

    class FakeProcessor:
        tokenizer = FakeProcessorTokenizer()

        def apply_chat_template(self, messages, **kwargs):
            assert messages[0] == {"role": "system", "content": HTML_GAME_SYSTEM_PROMPT}
            return "formatted prompt"

    adapter_path = Path(__file__).parents[3] / "scripts" / "html_game_dataset.py"
    dataset_cls = load_extern_object(str(adapter_path), "HtmlGameDataset")
    dataset = dataset_cls.__new__(dataset_cls)
    dataset.image_key = "images"
    dataset.video_key = "videos"
    dataset.audio_key = "audios"
    dataset.processor = FakeProcessor()
    dataset.tokenizer = None
    dataset.prompt_key = "prompt"
    dataset.filter_overlong_prompts = True
    dataset.apply_chat_template_kwargs = {}
    dataset.tool_schemas = None
    dataset.image_patch_size = 14
    dataset.config = {}
    dataset.max_prompt_length = 10
    dataset.mm_processor_kwargs = {}
    dataset.num_workers = 1
    dataset.serialize_dataset = True

    dataframe = Dataset.from_list([{"prompt": "Build a maze game."}])

    filtered = dataset.maybe_filter_out_long_prompts(dataframe)

    assert len(filtered) == 1


def test_source_indices_are_added_before_random_sample_selection():
    dataset = _mock_html_game_dataset()
    dataframe = Dataset.from_list(
        [
            {"prompt": "Build game zero."},
            {"prompt": "Build game one."},
            {"prompt": "Build game two."},
            {"prompt": "Build game three."},
        ]
    )

    prepared = dataset.prepare_dataframe_before_sample_selection(dataframe)
    sampled = prepared.select([3, 1])

    assert sampled[HtmlGameDataset._SOURCE_INDEX_KEY] == [3, 1]
