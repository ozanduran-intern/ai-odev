"""
장소 추천 도구
전 세계 도시별 + 카테고리별 장소 추천
주요 도시는 시뮬레이션 데이터, 기타 도시는 LLM 생성
"""

import random

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from config import MODEL_NAME, BASE_URL


# 주요 도시 시뮬레이션 데이터 (기존 한국 + 터키 + 세계 주요 도시)
PLACE_DATA: dict[str, dict[str, list[dict]]] = {
    "seoul": {
        "food": [
            {"name": "Gwangjang Market (광장시장)", "desc": "Traditional market paradise. Famous for bindaetteok, mayak-gimbap, and yukhoe."},
            {"name": "Euljiro Old Restaurant Street (을지로 노포거리)", "desc": "Decades-old eateries famous for gopchang and octopus dishes."},
            {"name": "Itaewon International Food Street (이태원)", "desc": "International cuisine from Turkish kebab to Mexican tacos."},
            {"name": "Tongin Market (통인시장)", "desc": "Unique lunch-box cafe where you pick dishes with old Korean coins. Local hidden gem."},
            {"name": "Mapo Jjeokgalbi Alley (마포 떡갈비골목)", "desc": "Locals-only grilled short rib patty street. Smoky, authentic, affordable."},
            {"name": "Noryangjin Fish Market (노량진 수산시장)", "desc": "Massive seafood market. Pick live fish, have it prepared upstairs instantly."},
        ],
        "attractions": [
            {"name": "Gyeongbokgung Palace (경복궁)", "desc": "Main royal palace of Joseon dynasty. Free entry in hanbok."},
            {"name": "N Seoul Tower (N서울타워)", "desc": "Panoramic city views from Namsan. Stunning night scenery."},
            {"name": "Bukchon Hanok Village (북촌 한옥마을)", "desc": "600-year-old traditional Korean houses. Popular photo spot."},
            {"name": "Ihwa Mural Village (이화벽화마을)", "desc": "Hidden hilltop village covered in street art. Quieter alternative to Bukchon."},
            {"name": "Seoullo 7017", "desc": "Elevated park on a former highway overpass. Urban oasis with sky gardens."},
            {"name": "Changdeokgung Secret Garden (창덕궁 비원)", "desc": "UNESCO site with a 300-year-old secret garden. Guided tours only, very intimate."},
        ],
        "shopping": [
            {"name": "Myeongdong (명동)", "desc": "Hub for cosmetics, fashion, and K-beauty. Great street food."},
            {"name": "Hongdae (홍대)", "desc": "Youthful shopping and indie culture. Vintage shops and indie brands."},
            {"name": "Namdaemun Market (남대문시장)", "desc": "Korea's largest traditional market. Affordable clothes and souvenirs."},
            {"name": "Garosu-gil (가로수길)", "desc": "Tree-lined boutique street in Gangnam. Designer shops and artisan cafes."},
            {"name": "Dongdaemun Design Plaza (DDP)", "desc": "Zaha Hadid landmark. Night flea markets and K-fashion wholesale."},
            {"name": "Ikseon-dong (익선동)", "desc": "Tiny hanok alley turned trendy. Indie boutiques, tea houses, vintage finds."},
        ],
        "nightlife": [
            {"name": "Gangnam (강남)", "desc": "Seoul's biggest nightlife district with clubs, bars, and karaoke."},
            {"name": "Itaewon (이태원)", "desc": "Foreigner-friendly bars and clubs. Rooftop bars with diverse vibes."},
            {"name": "Euljiro Hip Alley (을지로)", "desc": "Hidden retro bars in old factory buildings."},
            {"name": "Hapjeong / Mangwon (합정/망원)", "desc": "Indie bar scene. Craft cocktails in converted houses. Quiet but cool."},
            {"name": "Yeonnam-dong (연남동)", "desc": "Cozy wine bars and speakeasies near Gyeongui Line Forest Park."},
            {"name": "Jongno 3-ga (종로3가)", "desc": "Old-school Korean drinking culture. Pojangmacha tents and soju under the stars."},
        ],
    },
    "busan": {
        "food": [
            {"name": "Jagalchi Fish Market (자갈치시장)", "desc": "Korea's largest seafood market. Fresh sashimi on the spot."},
            {"name": "BIFF Square (BIFF광장)", "desc": "Home of ssiat-hotteok and Busan fish cakes. Iconic street food."},
            {"name": "Seomyeon Pork Soup Alley (서면 돼지국밥골목)", "desc": "Busan's soul food: pork soup with rice. 24-hour restaurants."},
            {"name": "Haeundae Traditional Market (해운대 전통시장)", "desc": "Hidden local market. Grilled clams, tteokbokki, and fresh juice."},
            {"name": "Gukje Market Food Alley (국제시장 먹자골목)", "desc": "Underground food court. Bindaetteok and kalguksu since the Korean War era."},
            {"name": "Millak Waterside Park Raw Fish (민락수변공원)", "desc": "Buy sashimi from the market, eat it at the park overlooking the ocean. Locals' secret."},
        ],
        "attractions": [
            {"name": "Haeundae Beach (해운대)", "desc": "Korea's most famous beach. Summer swimming, winter sunrise."},
            {"name": "Gamcheon Culture Village (감천문화마을)", "desc": "Colorful hillside village. Called 'Korea's Santorini'."},
            {"name": "Haedong Yonggungsa Temple (해동용궁사)", "desc": "Seaside cliff temple. Spectacular sunrise over the ocean."},
            {"name": "Taejongdae Park (태종대)", "desc": "Coastal cliff park with lighthouse. Dramatic ocean views and hiking trails."},
            {"name": "Huinnyeoul Culture Village (흰여울문화마을)", "desc": "Quiet seaside village on a cliff. Busan's hidden gem, way less crowded than Gamcheon."},
            {"name": "Oryukdo Skywalk (오륙도 스카이워크)", "desc": "Glass bridge over the ocean. Vertigo-inducing views where the sea meets the cliffs."},
        ],
        "shopping": [
            {"name": "Nampo-dong (남포동)", "desc": "Busan's main shopping street. Near Gukje and Kkangtong markets."},
            {"name": "Centum City (센텀시티)", "desc": "World's largest department store: Shinsegae Centum City."},
            {"name": "Gwangbok-ro (광복로)", "desc": "Fashion and souvenir street. Busan's Myeongdong."},
            {"name": "Jeonpo Cafe Street (전포카페거리)", "desc": "Converted warehouse district. Design shops, roasters, and indie bookstores."},
            {"name": "Gukje Market (국제시장)", "desc": "Historic wartime market. Vintage finds, fabrics, and handmade goods."},
            {"name": "F1963 (수영)", "desc": "Old wire factory turned cultural complex. Galleries, bookshop, and artisan market."},
        ],
        "nightlife": [
            {"name": "Seomyeon (서면)", "desc": "Busan's busiest nightlife area. Clubs, bars, and pojangmacha."},
            {"name": "Gwangalli Beach Bars (광안리)", "desc": "Beach bars with views of the lit-up Gwangan Bridge."},
            {"name": "Haeundae Pojangmacha (해운대 포차)", "desc": "Beach-side tent bars. Seafood snacks and soju."},
            {"name": "Kyungsung Univ. Area (경성대)", "desc": "Student nightlife hub. Cheap drinks, karaoke, and late-night food."},
            {"name": "The Bay 101 (더베이101)", "desc": "Yacht harbor lounge. Premium cocktails with Marine City skyline views."},
            {"name": "Nampo Rooftop Bars (남포)", "desc": "Hidden rooftop bars above the old downtown. Views of Busan Tower at night."},
        ],
    },
    "jeju": {
        "food": [
            {"name": "Jeju Black Pork Street (흑돼지거리)", "desc": "Jeju's specialty black pork BBQ. Rich, savory flavor."},
            {"name": "Dongmun Market (동문시장)", "desc": "Jeju's largest market. Seafood, hallabong juice, omegi-tteok."},
            {"name": "Haenyeo's House (해녀의 집)", "desc": "Fresh abalone and seafood caught by haenyeo divers."},
            {"name": "Dombe Pork (돔베고기)", "desc": "Jeju-only boiled pork dish. Served cold with kimchi. Unique island flavor."},
            {"name": "Hallim Seafood Village (한림 해물탕)", "desc": "Oceanfront restaurants serving massive seafood hotpots. Worth the drive."},
            {"name": "Jeju Mandarin Cafe Trail", "desc": "Cafes surrounded by mandarin orchards. Fresh-squeezed juice and tarts."},
        ],
        "attractions": [
            {"name": "Seongsan Ilchulbong (성산일출봉)", "desc": "UNESCO site. Volcanic crater famous for stunning sunrises."},
            {"name": "Hallasan Mountain (한라산)", "desc": "Korea's highest peak (1,947m). Beautiful in every season."},
            {"name": "Manjanggul Lava Tube (만장굴)", "desc": "One of the world's longest lava tubes."},
            {"name": "Udo Island (우도)", "desc": "Tiny island off Jeju's coast. Turquoise water, peanut ice cream, coral beaches."},
            {"name": "Seopjikoji (섭지코지)", "desc": "Dramatic coastal cliff walk. Glass church and canola flower fields."},
            {"name": "Camellia Hill (카멜리아힐)", "desc": "Lush botanical garden. 6,000 camellia trees and hidden forest paths."},
        ],
        "shopping": [
            {"name": "Jeju Jungang-ro (중앙로)", "desc": "Central shopping street. Souvenirs and local specialties."},
            {"name": "Seogwipo Olle Market (올레시장)", "desc": "Tangerines, chocolate, and handmade crafts."},
            {"name": "Jeju Art Markets", "desc": "Local artisan brands and handmade souvenirs."},
            {"name": "Innisfree Jeju House", "desc": "Flagship store on a mandarin farm. DIY cosmetics and organic cafe."},
            {"name": "Jeju Haenyeo Museum Shop", "desc": "Unique souvenirs about Jeju's diving women heritage. Nowhere else in the world."},
            {"name": "Woljeongri Beach Shops (월정리)", "desc": "Beachside boutiques. Surf gear, handmade jewelry, and Jeju crafts."},
        ],
        "nightlife": [
            {"name": "Tapdong (탑동)", "desc": "Coastal bars and cafes. Relaxed evening by the sea."},
            {"name": "Lee Jung-seop Street (이중섭거리)", "desc": "Cozy bars and gallery cafes on the artist street."},
            {"name": "Hyeopjae Beach Cafes (협재)", "desc": "Cafes overlooking emerald waters and sunset."},
            {"name": "Hamdeok Beach Bars (함덕)", "desc": "Barefoot beach bars. Bonfire nights and acoustic music in summer."},
            {"name": "Jeju Brewery Taproom", "desc": "Local craft brewery. Jeju Wit Ale brewed with island tangerines."},
            {"name": "Aewol Coastal Cafes (애월)", "desc": "Instagram-famous cliff cafes. Watch the sunset with Hallasan behind you."},
        ],
    },
    "istanbul": {
        "food": [
            {"name": "Karaköy Lokantası", "desc": "Modern Turkish cuisine in historic Karaköy. Excellent meze and seafood."},
            {"name": "Çiya Sofrası (Kadıköy)", "desc": "Legendary spot for authentic Anatolian dishes from all regions."},
            {"name": "Sultanahmet Köftecisi", "desc": "Famous since 1920 for the best köfte (meatballs) in Istanbul."},
            {"name": "Asmalımescit Meyhane Street", "desc": "Historic raki-and-meze alley. Live fasıl music, bohemian atmosphere."},
            {"name": "Balıkçı Sabahattin", "desc": "Old-school fish restaurant in a wooden Ottoman house. Family-run since 1927."},
            {"name": "Kadıköy Market (Tarihi Kadıköy Çarşısı)", "desc": "Asian side food market. Best simit, börek, and Turkish breakfast ingredients."},
        ],
        "attractions": [
            {"name": "Hagia Sophia (Ayasofya)", "desc": "Iconic 1500-year-old cathedral turned mosque. Architectural marvel."},
            {"name": "Grand Bazaar (Kapalıçarşı)", "desc": "One of the world's oldest covered markets with 4,000+ shops."},
            {"name": "Bosphorus Cruise", "desc": "Scenic boat ride between Europe and Asia. Stunning views."},
            {"name": "Basilica Cistern (Yerebatan Sarnıcı)", "desc": "Underground Byzantine water palace. Medusa heads, eerie lights, 1500 years old."},
            {"name": "Balat & Fener", "desc": "Colorful Greek/Jewish quarter. Rainbow houses, antique shops, best Instagram spots."},
            {"name": "Pierre Loti Hill (Eyüp)", "desc": "Hilltop cafe with panoramic Golden Horn views. Take the cable car up."},
        ],
        "shopping": [
            {"name": "Grand Bazaar (Kapalıçarşı)", "desc": "Historic covered market. Carpets, jewelry, ceramics, spices."},
            {"name": "İstiklal Caddesi", "desc": "Iconic pedestrian avenue. Fashion, bookshops, and street performers."},
            {"name": "Arasta Bazaar", "desc": "Quiet alternative near Blue Mosque. Handcrafted goods."},
            {"name": "Çukurcuma (Beyoğlu)", "desc": "Antique dealers and vintage shops. Hidden treasures in every corner."},
            {"name": "Mısır Çarşısı (Spice Bazaar)", "desc": "Egyptian bazaar. Turkish delight, saffron, dried fruits, and souvenirs."},
            {"name": "Nişantaşı", "desc": "Istanbul's luxury fashion district. Turkish and international designers."},
        ],
        "nightlife": [
            {"name": "Beyoğlu / Taksim", "desc": "Heart of Istanbul nightlife. Bars, clubs, and live music venues."},
            {"name": "Kadıköy Barlar Sokağı", "desc": "Asian side bar street. Craft beer and indie music scene."},
            {"name": "Karaköy", "desc": "Trendy waterfront area with rooftop bars and cocktail lounges."},
            {"name": "Arnavutköy Sahili", "desc": "Bosphorus-side fish restaurants. Quiet raki by the water. Romantic."},
            {"name": "Balat Cafes & Bars", "desc": "Bohemian nightlife in colorful streets. Wine bars in converted houses."},
            {"name": "Bebek Sahili", "desc": "Upscale waterfront. Late-night tea and dessert with Bosphorus views."},
        ],
    },
    "tokyo": {
        "food": [
            {"name": "Tsukiji Outer Market", "desc": "Fresh sushi, seafood, and Japanese street food paradise."},
            {"name": "Ramen Street (Tokyo Station)", "desc": "Eight top ramen shops under Tokyo Station."},
            {"name": "Omoide Yokocho (Shinjuku)", "desc": "Atmospheric alley of tiny yakitori and izakaya joints."},
            {"name": "Hoppy Street (Asakusa)", "desc": "Open-air izakaya strip. Locals drink hoppy beer with beef stew. Very retro."},
            {"name": "Yanaka Ginza", "desc": "Old Tokyo shopping street. Croquettes, cat-shaped sweets, and shaved ice."},
            {"name": "Kappabashi Street", "desc": "Kitchen town — fake food samples, knives, and ceramics. Foodie souvenir heaven."},
        ],
        "attractions": [
            {"name": "Senso-ji Temple (Asakusa)", "desc": "Tokyo's oldest temple. Iconic Kaminarimon gate."},
            {"name": "Shibuya Crossing", "desc": "World's busiest pedestrian crossing. Iconic Tokyo experience."},
            {"name": "Meiji Shrine", "desc": "Peaceful Shinto shrine in a forest, heart of Harajuku."},
            {"name": "teamLab Borderless", "desc": "Immersive digital art museum. Walk through infinite light installations."},
            {"name": "Shimokitazawa", "desc": "Bohemian neighborhood. Vintage shops, tiny theaters, and indie cafes."},
            {"name": "Yanaka Cemetery & Old Town", "desc": "Edo-era atmosphere survived the war. Cherry blossoms, temples, stray cats."},
        ],
        "shopping": [
            {"name": "Harajuku / Takeshita Street", "desc": "Youth fashion, kawaii culture, and unique boutiques."},
            {"name": "Akihabara", "desc": "Electronics, anime, manga, and otaku culture hub."},
            {"name": "Ginza", "desc": "Luxury shopping district. High-end brands and department stores."},
            {"name": "Nakameguro", "desc": "Canal-side boutiques and designer concept stores. Tokyo's cool-kid district."},
            {"name": "Shimokitazawa Vintage", "desc": "Best vintage clothing in Tokyo. Dozens of curated secondhand shops."},
            {"name": "Daikanyama T-Site", "desc": "Beautiful bookstore complex. Design, art, and lifestyle shopping."},
        ],
        "nightlife": [
            {"name": "Shinjuku Golden Gai", "desc": "Maze of 200+ tiny bars. Each with unique character."},
            {"name": "Roppongi", "desc": "International nightlife district. Clubs and late-night dining."},
            {"name": "Shibuya", "desc": "Trendy bars, clubs, and izakayas for the young crowd."},
            {"name": "Nakameguro", "desc": "Quiet cocktail bars along the canal. Intimate, stylish, no crowds."},
            {"name": "Sangenjaya", "desc": "Local favorite. Tiny standing bars and sake spots. Zero tourists."},
            {"name": "Shimokitazawa", "desc": "Live music venues, craft beer bars, and late-night curry."},
        ],
    },
    "paris": {
        "food": [
            {"name": "Le Marais District", "desc": "Trendy neighborhood with bistros, falafel, and patisseries."},
            {"name": "Rue Montorgueil", "desc": "Historic food street. Bakeries, cheese shops, and cafes."},
            {"name": "Saint-Germain-des-Prés", "desc": "Classic Parisian cafes and fine dining restaurants."},
            {"name": "Marché des Enfants Rouges", "desc": "Oldest covered market in Paris. Moroccan couscous, Japanese bento, French crêpes."},
            {"name": "Canal Saint-Martin", "desc": "Hipster picnic culture. Grab cheese and wine, sit by the canal like a local."},
            {"name": "Rue Sainte-Anne (Little Tokyo)", "desc": "Best ramen and udon in Europe, hidden in the 1st arrondissement. Locals-only vibe."},
        ],
        "attractions": [
            {"name": "Eiffel Tower", "desc": "The iconic symbol of Paris. Best views at sunset."},
            {"name": "Louvre Museum", "desc": "World's largest art museum. Home of the Mona Lisa."},
            {"name": "Montmartre & Sacré-Cœur", "desc": "Artistic hilltop village with stunning city views."},
            {"name": "Musée d'Orsay", "desc": "Impressionist masterpieces in a converted train station. Less crowded than Louvre."},
            {"name": "Père Lachaise Cemetery", "desc": "Hauntingly beautiful. Jim Morrison, Oscar Wilde, Chopin rest here."},
            {"name": "La Petite Ceinture", "desc": "Abandoned railway turned secret garden. Walk where trains once ran through Paris."},
        ],
        "shopping": [
            {"name": "Champs-Élysées", "desc": "Famous avenue with luxury brands and flagship stores."},
            {"name": "Le Marais", "desc": "Trendy boutiques, vintage shops, and local designers."},
            {"name": "Galeries Lafayette", "desc": "Iconic department store with stunning art nouveau dome."},
            {"name": "Saint-Ouen Flea Market", "desc": "World's largest antique market. Vintage treasures from every era."},
            {"name": "Merci Concept Store", "desc": "Curated lifestyle store in a converted factory. Profits go to charity."},
            {"name": "Rue de Rivoli Bookstalls", "desc": "Seine-side bouquinistes. Vintage posters, old books, and Paris memorabilia."},
        ],
        "nightlife": [
            {"name": "Oberkampf / Ménilmontant", "desc": "Hipster bars and live music. Authentic Parisian nightlife."},
            {"name": "Pigalle / SoPi", "desc": "Cocktail bars, cabarets, and the famous Moulin Rouge area."},
            {"name": "Le Marais", "desc": "Trendy bars, LGBTQ+ scene, and late-night bistros."},
            {"name": "Belleville", "desc": "Multicultural nightlife. Rooftop views, cheap wine bars, and street art."},
            {"name": "Seine River Banks", "desc": "Summer pop-up bars along the river. Dance under the bridges."},
            {"name": "Le Comptoir Général", "desc": "Hidden courtyard bar in a former warehouse. Tropical decor, DJ sets, charity events."},
        ],
    },
    "bangkok": {
        "food": [
            {"name": "Yaowarat (Chinatown)", "desc": "Best street food in Bangkok. Pad thai, seafood, mango sticky rice."},
            {"name": "Or Tor Kor Market", "desc": "Premium fresh market. Tropical fruits and Thai curries."},
            {"name": "Khao San Road", "desc": "Backpacker hub with cheap eats and vibrant atmosphere."},
            {"name": "Bang Rak (Charoen Krung)", "desc": "Old Bangkok food district. Michelin-starred street stalls. Jay Fai lives here."},
            {"name": "Victory Monument Boat Noodles", "desc": "Tiny bowls, massive flavor. Eat 10+ bowls for $5. Local speed-eating tradition."},
            {"name": "Ari Neighborhood", "desc": "Trendy local district. Craft coffee, Thai fusion, and dessert cafes. Zero tourists."},
        ],
        "attractions": [
            {"name": "Grand Palace & Wat Phra Kaew", "desc": "Thailand's most sacred Buddhist temple and royal palace."},
            {"name": "Wat Arun", "desc": "Temple of Dawn. Stunning riverside temple with ornate spires."},
            {"name": "Chatuchak Weekend Market", "desc": "Massive market with 15,000+ stalls. Everything you can imagine."},
            {"name": "Jim Thompson House", "desc": "Silk merchant's teak mansion. Museum of Thai art hidden in a garden."},
            {"name": "Khlong Lat Mayom Floating Market", "desc": "Real floating market (not touristy). Weekend only. Boat noodles on the water."},
            {"name": "Bang Krachao (Green Lung)", "desc": "Jungle island in the middle of Bangkok. Rent a bike, ride through mangroves."},
        ],
        "shopping": [
            {"name": "Chatuchak Weekend Market", "desc": "One of the world's largest outdoor markets."},
            {"name": "Siam Paragon / CentralWorld", "desc": "Luxury malls in the heart of Bangkok."},
            {"name": "Asiatique The Riverfront", "desc": "Night market by the river with shops and restaurants."},
            {"name": "Sampeng Lane (Chinatown)", "desc": "Narrow alley of wholesale goods. Fabric, accessories, and trinkets for pennies."},
            {"name": "JJ Green Night Market", "desc": "Vintage clothing, retro collectibles, and live music. Young local crowd."},
            {"name": "Icon Siam", "desc": "Riverside mega-mall with an indoor floating market. Modern meets traditional."},
        ],
        "nightlife": [
            {"name": "Khao San Road", "desc": "Backpacker party street. Cheap drinks and street performances."},
            {"name": "Thonglor / Ekkamai", "desc": "Trendy bars, rooftop lounges, and craft cocktails."},
            {"name": "RCA (Royal City Avenue)", "desc": "Club district popular with locals."},
            {"name": "Vertigo Rooftop Bar (Banyan Tree)", "desc": "61st floor open-air bar. Jaw-dropping 360° city views."},
            {"name": "Tep Bar (Charoen Krung)", "desc": "Traditional Thai music meets craft cocktails. Infused ya dong shots."},
            {"name": "Talad Rot Fai (Train Market)", "desc": "Night market with live bands, vintage cars, and open-air bars. Local party scene."},
        ],
    },
}

# 도시명 alias 매핑
CITY_ALIASES: dict[str, str] = {
    "서울": "seoul", "부산": "busan", "제주": "jeju", "제주도": "jeju",
    "이스탄불": "istanbul", "도쿄": "tokyo", "동경": "tokyo",
    "파리": "paris", "방콕": "bangkok", "런던": "london",
    "뉴욕": "new york", "로마": "rome",
}

SUPPORTED_CATEGORIES = "food, attractions, shopping, nightlife"


def _llm_recommend(city: str, category: str) -> str:
    """DB에 없는 도시는 LLM으로 추천 생성"""
    llm = ChatOllama(model=MODEL_NAME, base_url=BASE_URL, temperature=0.7)

    messages = [
        {"role": "system", "content": (
            f"You are a travel expert who knows hidden gems. Recommend exactly 3 {category} places in {city}. "
            f"Mix 1 popular spot with 2 lesser-known local favorites. "
            f"For each place, give the name and a brief reason WHY it's special (not just what it is). "
            f"Format:\n1. Name - Description\n2. Name - Description\n3. Name - Description\n"
            f"Be specific, opinionated, and helpful. No generic descriptions."
        )},
        {"role": "user", "content": f"Top 3 {category} in {city} — mix popular and hidden gems"},
    ]

    response = llm.invoke(messages)
    return response.content.strip()


@tool
def recommend_places(city: str, category: str) -> str:
    """Recommend places in any city worldwide. Categories: food, attractions, shopping, nightlife.
    Works for any city — Seoul, Istanbul, Tokyo, Paris, Bangkok, New York, and more.
    Example: recommend_places('paris', 'food')"""
    try:
        city_key = city.strip().lower()
        city_key = CITY_ALIASES.get(city_key, city_key)
        cat_key = category.strip().lower()

        if cat_key not in ("food", "attractions", "shopping", "nightlife"):
            return f"Unknown category '{category}'. Supported: {SUPPORTED_CATEGORIES}"

        city_display = city.strip().title()

        # DB에 있으면 시뮬레이션 데이터에서 랜덤 3개 선택
        if city_key in PLACE_DATA and cat_key in PLACE_DATA[city_key]:
            all_places = PLACE_DATA[city_key][cat_key]
            selected = random.sample(all_places, min(3, len(all_places)))
            result = f"📍 {city_display} - {cat_key.upper()} TOP 3:\n\n"
            for i, place in enumerate(selected, 1):
                result += f"  {i}. {place['name']}\n     {place['desc']}\n\n"
            return result.rstrip()

        # DB에 없으면 LLM으로 생성
        llm_result = _llm_recommend(city_display, cat_key)
        return f"📍 {city_display} - {cat_key.upper()} TOP 3:\n\n{llm_result}"

    except Exception as e:
        return f"Place recommendation error: {str(e)}"
