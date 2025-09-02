class HotlineManager:
    def __init__(self):
        self.hotlines = {
            "qtmac": "0901 941 941",
            "npa_quang_binh": "1800 1741",
            "npa_quang_tri": "0901 941 941",
            "npa_thua_thien_hue": "0988 796 120"
        }

    def get_hotline(self, location: str) -> str:
        # Normalize key
        key = location.lower().replace(" ", "_")
        return self.hotlines.get(key, "Xin lỗi, chưa có số hotline cho khu vực này.")
