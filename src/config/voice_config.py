from collections import defaultdict


class VoiceConfig:
    def __init__(self, voice_name: str):
        self.voice_name = voice_name


voice_list = defaultdict(lambda: VoiceConfig("en-US-AriaNeural"))

voice_dict = {
    "male": VoiceConfig("en-US-GuyNeural"),
    "female": VoiceConfig("en-US-AriaNeural"),
}

for key, value in voice_dict.items():
    voice_list[key] = value

__all__ = ["voice_list", "VoiceConfig"]
