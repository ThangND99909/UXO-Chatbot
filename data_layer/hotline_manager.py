class HotlineManager:
    def __init__(self):
        self.hotlines = {
            # Key chính
            "qtmac": "0901 941 941",
            "npa_quang_binh": "1800 1741",
            "npa_quang_tri": "0901 941 941", 
            "npa_thua_thien_hue": "0988 796 120",
            
            # Thêm các key alternative
            "quang_binh": "1800 1741",
            "qb": "1800 1741",
            "quang_tri": "0901 941 941",
            "qt": "0901 941 941",
            "thua_thien_hue": "0988 796 120",
            "hue": "0988 796 120",
            "tth": "0988 796 120",
            
            # Thêm các biến thể có dấu và không dấu
            "quảng_bình": "1800 1741",
            "quảng_trị": "0901 941 941",
            "thừa_thiên_huế": "0988 796 120"
        }

    def get_hotline(self, location: str) -> str:
        # Normalize key - xử lý nhiều trường hợp hơn
        key = location.lower().replace(" ", "_").replace("-", "_").replace(".", "")
        print(f"🔍 Looking up hotline for key: '{key}'")
        
        # Kiểm tra trực tiếp
        if key in self.hotlines:
            return self.hotlines[key]
        
        # Fallback: kiểm tra partial match
        for hotline_key, number in self.hotlines.items():
            if hotline_key in key or key in hotline_key:
                return number
        
        return "Xin lỗi, chưa có số hotline cho khu vực này."