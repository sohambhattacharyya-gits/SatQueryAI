class QueryRouter:
    def route_query(self, query_text, number_of_images, mode="auto"):
        query_text = query_text.lower()
        
        if number_of_images == 2:
            if mode == "cross_modal" or any(w in query_text for w in ["sar", "radar", "optical", "fusion", "together"]):
                return "optical_sar_fusion"
            return "change_detection"
            
        if any(word in query_text for word in ["where", "locate", "find", "point", "highlight", "box"]):
            return "grounding"
            
        if any(word in query_text for word in ["describe", "caption", "tell me about"]):
            return "captioning"
            
        return "vqa"
