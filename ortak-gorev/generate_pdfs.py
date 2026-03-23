"""PDF 학습 자료 생성 스크립트 - 한글 TTF 폰트 사용"""

import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors

# 한글 + 터키어 + 영어 모두 지원하는 폰트 등록
ARIAL_UNICODE = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
pdfmetrics.registerFont(TTFont("ArialUnicode", ARIAL_UNICODE))

FONT = "ArialUnicode"

BLUE = HexColor("#1a5276")
DARK = HexColor("#2c3e50")
ACCENT = HexColor("#2980b9")
GREEN = HexColor("#1e8449")
TABLE_HEADER_BG = HexColor("#2980b9")
TABLE_ALT_BG = HexColor("#eaf2f8")


def S(name, **kw):
    """ParagraphStyle 헬퍼"""
    defaults = {"fontName": FONT, "textColor": DARK, "leading": 15}
    defaults.update(kw)
    return ParagraphStyle(name, **defaults)


STYLES = {
    "title": S("T", fontSize=22, textColor=BLUE, spaceAfter=12, alignment=1),
    "subtitle": S("Sub", fontSize=11, textColor=DARK, spaceAfter=4, alignment=1),
    "h1": S("H1", fontSize=16, textColor=BLUE, spaceBefore=18, spaceAfter=8),
    "h2": S("H2", fontSize=13, textColor=ACCENT, spaceBefore=12, spaceAfter=6),
    "body": S("B", fontSize=10, leading=15),
    "bullet": S("Bul", fontSize=10, leading=16, leftIndent=20, bulletIndent=10),
    "example": S("Ex", fontSize=10, leading=16, textColor=GREEN, leftIndent=20),
    "note": S("N", fontSize=9, leading=13, textColor=HexColor("#7f8c8d"), leftIndent=20),
}


def hr():
    return HRFlowable(width="100%", thickness=0.5, color=ACCENT, spaceAfter=8, spaceBefore=8)


def sp(h=6):
    return Spacer(1, h)


def P(text, style="body"):
    return Paragraph(text, STYLES[style])


def make_table(headers, rows, col_widths=None):
    data = [headers] + rows
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), TABLE_HEADER_BG),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, -1), FONT),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, TABLE_ALT_BG]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


# ═══════════════════════════════════════════════
# PDF 1: Korean Grammar
# ═══════════════════════════════════════════════
def build_grammar_pdf(path):
    doc = SimpleDocTemplate(path, pagesize=A4,
                            topMargin=20*mm, bottomMargin=20*mm,
                            leftMargin=20*mm, rightMargin=20*mm)
    story = []

    # Title
    story += [
        P("Korean Grammar Basics", "title"),
        P("한국어 문법 기초 / Essential Grammar for Korean Learners", "subtitle"),
        hr(), sp(10),
    ]

    # ── 1. Sentence Structure ──
    story += [
        P("1. Korean Sentence Structure (SOV) / 한국어 문장 구조", "h1"),
        P("Korean follows a <b>Subject-Object-Verb (SOV)</b> word order, which is different from "
          "English (SVO). The verb always comes at the end of the sentence."),
        sp(),
        P("English (SVO): I eat rice."),
        P("Korean (SOV): 나는 + 밥을 + 먹어요 (I + rice + eat)"),
        sp(),
    ]
    story.append(make_table(
        ["English", "Korean (한국어)", "Literal Order"],
        [
            ["I eat rice.", "나는 밥을 먹어요.", "I + rice-obj + eat"],
            ["She reads a book.", "그녀는 책을 읽어요.", "She + book-obj + read"],
            ["We go to school.", "우리는 학교에 가요.", "We + school-to + go"],
        ],
        col_widths=[50*mm, 55*mm, 55*mm],
    ))
    story += [
        sp(),
        P("<b>Key Point:</b> Even though word order is flexible in Korean, the verb must "
          "always remain at the end. Particles mark the grammatical role of each word, "
          "so changing the order does not change the meaning as dramatically as in English.", "note"),
        sp(12),
    ]

    # ── 2. Particles ──
    story += [
        P("2. Korean Particles / 한국어 조사", "h1"),
        P("Particles are attached directly to nouns to indicate their grammatical function. "
          "They are one of the most important features of Korean grammar."),
        sp(),

        P("2.1 Topic Markers: 은/는", "h2"),
        P("은 (eun) is used after consonant-ending nouns, 는 (neun) after vowel-ending nouns. "
          "They mark the <b>topic</b> of the sentence — what the sentence is about."),
        sp(),
        P("• 나<b>는</b> 학생이에요. — As for me, I am a student.", "bullet"),
        P("• 오늘<b>은</b> 날씨가 좋아요. — As for today, the weather is good.", "bullet"),
        sp(),

        P("2.2 Subject Markers: 이/가", "h2"),
        P("이 (i) is used after consonant-ending nouns, 가 (ga) after vowel-ending nouns. "
          "They mark the <b>subject</b> performing the action."),
        sp(),
        P("• 비<b>가</b> 와요. — Rain comes. (It's raining.)", "bullet"),
        P("• 친구<b>가</b> 와요. — A friend comes.", "bullet"),
        P("• 물<b>이</b> 차가워요. — The water is cold.", "bullet"),
        sp(),

        P("2.3 Object Markers: 을/를", "h2"),
        P("을 (eul) is used after consonant-ending nouns, 를 (reul) after vowel-ending nouns. "
          "They mark the <b>direct object</b> of the verb."),
        sp(),
        P("• 커피<b>를</b> 마셔요. — I drink coffee.", "bullet"),
        P("• 밥<b>을</b> 먹어요. — I eat rice.", "bullet"),
        sp(),

        P("2.4 Topic (은/는) vs Subject (이/가) — What's the Difference?", "h2"),
        P("This is one of the trickiest parts of Korean grammar for learners:"),
        sp(),
        P("• 은/는 introduces a topic or contrasts: 나<b>는</b> 학생이에요. (As for me, I'm a student.)", "bullet"),
        P("• 이/가 introduces new info or emphasizes: 비<b>가</b> 와요. (It's raining! — new info)", "bullet"),
        P("• Compare: 누가 왔어요? (Who came?) → 친구<b>가</b> 왔어요. (A friend came. — answering who)", "bullet"),
        sp(),

        P("2.5 Location &amp; Direction Particles", "h2"),
        P("• 에 (e) — at, in, on, to (static location or destination)", "bullet"),
        P("• 에서 (eseo) — at, in, from (where action occurs)", "bullet"),
        P("• (으)로 (euro/ro) — toward, by means of", "bullet"),
        sp(),
        P("• 학교<b>에</b> 가요. — I go to school. (destination)", "bullet"),
        P("• 학교<b>에서</b> 공부해요. — I study at school. (action location)", "bullet"),
        P("• 버스<b>로</b> 가요. — I go by bus. (means)", "bullet"),
        sp(12),
    ]

    # ── 3. Verb Conjugation ──
    story += [
        P("3. Verb Conjugation Basics / 동사 활용", "h1"),
        P("All Korean verbs end in 다 (da) in their dictionary form. To conjugate, "
          "remove 다 and add the appropriate ending."),
        sp(),
        P("3.1 Common Verb Stems", "h2"),
    ]
    story.append(make_table(
        ["Dictionary Form", "Meaning", "Stem"],
        [
            ["먹다", "to eat", "먹"],
            ["가다", "to go", "가"],
            ["보다", "to see", "보"],
            ["하다", "to do", "하"],
            ["오다", "to come", "오"],
            ["읽다", "to read", "읽"],
            ["쓰다", "to write", "쓰"],
            ["말하다", "to speak", "말하"],
        ],
        col_widths=[50*mm, 50*mm, 50*mm],
    ))
    story += [
        sp(),
        P("3.2 Polite Informal Ending (아요/어요)", "h2"),
        P("The most commonly used speech level. If the last vowel of the stem is "
          "ㅏ or ㅗ, add 아요. Otherwise, add 어요. 하다 becomes 해요 (special case)."),
        sp(),
    ]
    story.append(make_table(
        ["Dictionary", "Conjugation Rule", "Polite Form"],
        [
            ["먹다", "먹 + 어요 (last vowel ㅓ)", "먹어요"],
            ["가다", "가 + 아요 → contracts", "가요"],
            ["보다", "보 + 아요 → contracts", "봐요"],
            ["하다", "special case", "해요"],
            ["마시다", "마시 + 어요 → contracts", "마셔요"],
        ],
        col_widths=[45*mm, 60*mm, 50*mm],
    ))
    story += [sp(12)]

    # ── 4. Speech Levels ──
    story += [
        PageBreak(),
        P("4. Speech Levels (Formal &amp; Informal) / 말투", "h1"),
        P("Korean has multiple speech levels that express the speaker's relationship "
          "to the listener. The two most important levels for beginners:"),
        sp(),

        P("4.1 Formal Polite (합니다 style)", "h2"),
        P("Used in formal settings: news, presentations, meetings, speaking to elders "
          "you don't know well."),
        sp(),
    ]
    story.append(make_table(
        ["Type", "Ending", "Example"],
        [
            ["Statement", "-습니다/-ㅂ니다", "먹습니다. (I eat.)"],
            ["Question", "-습니까/-ㅂ니까", "어디 가십니까? (Where are you going?)"],
            ["Thank you", "", "감사합니다. (Thank you.)"],
        ],
        col_widths=[40*mm, 50*mm, 65*mm],
    ))

    story += [
        sp(),
        P("4.2 Polite Informal (해요 style)", "h2"),
        P("Used in everyday conversation. This is the most versatile and commonly used level."),
        sp(),
    ]
    story.append(make_table(
        ["Type", "Example", "Translation"],
        [
            ["Statement", "먹어요.", "I eat."],
            ["Question", "어디 가요?", "Where are you going?"],
            ["Thank you", "감사해요. / 고마워요.", "Thank you."],
        ],
        col_widths=[40*mm, 55*mm, 60*mm],
    ))

    story += [
        sp(),
        P("4.3 Casual / Intimate (해 style)", "h2"),
        P("Used with close friends of the same age or younger, children, or in very "
          "casual settings. <b>Do not use with strangers or elders.</b>"),
        sp(),
    ]
    story.append(make_table(
        ["Type", "Example", "Translation"],
        [
            ["Statement", "먹어.", "I eat."],
            ["Question", "어디 가?", "Where are you going?"],
            ["Thanks", "고마워.", "Thanks."],
        ],
        col_widths=[40*mm, 55*mm, 60*mm],
    ))

    # ── 5. Tenses ──
    story += [
        sp(12),
        P("5. Tenses: Past, Present, Future / 시제", "h1"),
        sp(),

        P("5.1 Present Tense / 현재 시제", "h2"),
        P("Use the basic conjugated form. It covers habitual and current actions."),
        P("• 나는 한국어를 공부해요. — I study Korean.", "bullet"),
        P("• 매일 커피를 마셔요. — I drink coffee every day.", "bullet"),
        sp(),

        P("5.2 Past Tense / 과거 시제", "h2"),
        P("Add 았/었 to the verb stem. If the last vowel is ㅏ or ㅗ, use 았어요. Otherwise, 었어요."),
        sp(),

        P("5.3 Future Tense / 미래 시제", "h2"),
        P("Add -(으)ㄹ 거예요 to the stem for plans/intentions."),
        sp(),

        P("5.4 Tense Summary Table / 시제 요약", "h2"),
    ]
    story.append(make_table(
        ["Verb (동사)", "Present (현재)", "Past (과거)", "Future (미래)"],
        [
            ["먹다 (eat)", "먹어요", "먹었어요", "먹을 거예요"],
            ["가다 (go)", "가요", "갔어요", "갈 거예요"],
            ["하다 (do)", "해요", "했어요", "할 거예요"],
            ["보다 (see)", "봐요", "봤어요", "볼 거예요"],
            ["오다 (come)", "와요", "왔어요", "올 거예요"],
            ["마시다 (drink)", "마셔요", "마셨어요", "마실 거예요"],
        ],
        col_widths=[38*mm, 38*mm, 38*mm, 45*mm],
    ))

    # ── 6. Negation ──
    story += [
        sp(12),
        P("6. Negation / 부정문", "h1"),
        P("There are two common ways to make negative sentences:"),
        sp(),

        P("6.1 Short Negation: 안 + verb", "h2"),
        P("• 안 먹어요. — I don't eat.", "bullet"),
        P("• 안 가요. — I don't go.", "bullet"),
        sp(),

        P("6.2 Long Negation: verb stem + 지 않아요", "h2"),
        P("• 먹지 않아요. — I don't eat.", "bullet"),
        P("• 가지 않아요. — I don't go.", "bullet"),
        sp(),

        P("<b>Tip:</b> Short negation is more conversational; long negation is slightly "
          "more formal or emphatic. Both are correct and interchangeable in most cases.", "note"),
        sp(),

        P("6.3 못 (Cannot) — Inability Negation", "h2"),
        P("• 못 먹어요. — I can't eat.", "bullet"),
        P("• 못 가요. — I can't go.", "bullet"),
        P("• 한국어를 못 해요. — I can't speak Korean.", "bullet"),
        sp(),

        P("6.4 Negation Summary Table", "h2"),
    ]
    story.append(make_table(
        ["Type", "Pattern", "Example", "Meaning"],
        [
            ["Don't (short)", "안 + verb", "안 먹어요", "I don't eat"],
            ["Don't (long)", "stem + 지 않아요", "먹지 않아요", "I don't eat"],
            ["Can't (short)", "못 + verb", "못 가요", "I can't go"],
            ["Can't (long)", "stem + 지 못해요", "가지 못해요", "I can't go"],
        ],
        col_widths=[33*mm, 40*mm, 40*mm, 42*mm],
    ))

    doc.build(story)
    print(f"✅ Created {path}")


# ═══════════════════════════════════════════════
# PDF 2: Korean Vocabulary
# ═══════════════════════════════════════════════
def build_vocabulary_pdf(path):
    doc = SimpleDocTemplate(path, pagesize=A4,
                            topMargin=20*mm, bottomMargin=20*mm,
                            leftMargin=20*mm, rightMargin=20*mm)
    story = []

    story += [
        P("Essential Korean Vocabulary", "title"),
        P("필수 한국어 어휘 / Core Words and Phrases for Beginners", "subtitle"),
        hr(), sp(10),
    ]

    # ── 1. Greetings ──
    story += [P("1. Greetings / 인사말", "h1")]
    story.append(make_table(
        ["Korean (한국어)", "English", "Türkçe", "Usage"],
        [
            ["안녕하세요", "Hello", "Merhaba", "Polite, universal"],
            ["안녕히 가세요", "Goodbye (to leaver)", "Hoşça kalın", "Said by person staying"],
            ["안녕히 계세요", "Goodbye (to stayer)", "Güle güle", "Said by person leaving"],
            ["감사합니다", "Thank you", "Teşekkür ederim", "Formal"],
            ["고마워요", "Thank you", "Teşekkürler", "Polite informal"],
            ["죄송합니다", "I'm sorry", "Özür dilerim", "Formal apology"],
            ["실례합니다", "Excuse me", "Pardon", "Getting attention"],
            ["네", "Yes", "Evet", "Polite"],
            ["아니요", "No", "Hayır", "Polite"],
            ["반갑습니다", "Nice to meet you", "Memnun oldum", "First meetings"],
            ["잘 지내셨어요?", "How have you been?", "Nasılsınız?", "Polite greeting"],
        ],
        col_widths=[38*mm, 38*mm, 38*mm, 42*mm],
    ))

    # ── 2. Numbers ──
    story += [
        sp(12),
        P("2. Numbers / 숫자", "h1"),
        P("Korean has TWO number systems: Sino-Korean (Chinese origin) and Native Korean."),
        sp(),
        P("2.1 Sino-Korean Numbers (used for dates, money, phone numbers, minutes)", "h2"),
    ]
    story.append(make_table(
        ["Number", "Korean (한국어)", "English", "Türkçe"],
        [
            ["0", "영", "zero", "sıfır"],
            ["1", "일", "one", "bir"],
            ["2", "이", "two", "iki"],
            ["3", "삼", "three", "üç"],
            ["4", "사", "four", "dört"],
            ["5", "오", "five", "beş"],
            ["6", "육", "six", "altı"],
            ["7", "칠", "seven", "yedi"],
            ["8", "팔", "eight", "sekiz"],
            ["9", "구", "nine", "dokuz"],
            ["10", "십", "ten", "on"],
            ["100", "백", "hundred", "yüz"],
            ["1000", "천", "thousand", "bin"],
        ],
        col_widths=[25*mm, 45*mm, 40*mm, 40*mm],
    ))

    story += [
        sp(),
        P("2.2 Native Korean Numbers (used for hours, counting, age)", "h2"),
    ]
    story.append(make_table(
        ["Number", "Korean (한국어)", "English", "Türkçe"],
        [
            ["1", "하나", "one", "bir"],
            ["2", "둘", "two", "iki"],
            ["3", "셋", "three", "üç"],
            ["4", "넷", "four", "dört"],
            ["5", "다섯", "five", "beş"],
            ["6", "여섯", "six", "altı"],
            ["7", "일곱", "seven", "yedi"],
            ["8", "여덟", "eight", "sekiz"],
            ["9", "아홉", "nine", "dokuz"],
            ["10", "열", "ten", "on"],
        ],
        col_widths=[25*mm, 45*mm, 40*mm, 40*mm],
    ))

    # ── 3. Days of the Week ──
    story += [
        sp(12), PageBreak(),
        P("3. Days of the Week / 요일", "h1"),
    ]
    story.append(make_table(
        ["Korean (한국어)", "English", "Türkçe", "Hanja Origin"],
        [
            ["월요일", "Monday", "Pazartesi", "Moon (月)"],
            ["화요일", "Tuesday", "Salı", "Fire (火)"],
            ["수요일", "Wednesday", "Çarşamba", "Water (水)"],
            ["목요일", "Thursday", "Perşembe", "Wood (木)"],
            ["금요일", "Friday", "Cuma", "Gold (金)"],
            ["토요일", "Saturday", "Cumartesi", "Earth (土)"],
            ["일요일", "Sunday", "Pazar", "Sun (日)"],
        ],
        col_widths=[38*mm, 35*mm, 35*mm, 42*mm],
    ))

    # ── 4. Colors ──
    story += [
        sp(12),
        P("4. Colors / 색깔", "h1"),
    ]
    story.append(make_table(
        ["Korean (한국어)", "English", "Türkçe"],
        [
            ["빨간색", "Red", "Kırmızı"],
            ["파란색", "Blue", "Mavi"],
            ["노란색", "Yellow", "Sarı"],
            ["초록색", "Green", "Yeşil"],
            ["하얀색 / 흰색", "White", "Beyaz"],
            ["검은색 / 검정색", "Black", "Siyah"],
            ["분홍색", "Pink", "Pembe"],
            ["보라색", "Purple", "Mor"],
            ["주황색", "Orange", "Turuncu"],
            ["갈색", "Brown", "Kahverengi"],
            ["회색", "Gray", "Gri"],
        ],
        col_widths=[52*mm, 52*mm, 52*mm],
    ))

    # ── 5. Common Verbs ──
    story += [
        sp(12),
        P("5. Common Verbs / 동사", "h1"),
    ]
    story.append(make_table(
        ["Korean (사전형)", "English", "Türkçe", "Polite Form"],
        [
            ["먹다", "to eat", "yemek", "먹어요"],
            ["마시다", "to drink", "içmek", "마셔요"],
            ["가다", "to go", "gitmek", "가요"],
            ["오다", "to come", "gelmek", "와요"],
            ["보다", "to see/watch", "görmek/izlemek", "봐요"],
            ["읽다", "to read", "okumak", "읽어요"],
            ["쓰다", "to write", "yazmak", "써요"],
            ["말하다", "to speak", "konuşmak", "말해요"],
            ["듣다", "to listen", "dinlemek", "들어요"],
            ["사다", "to buy", "satın almak", "사요"],
            ["자다", "to sleep", "uyumak", "자요"],
            ["일어나다", "to wake up", "uyanmak", "일어나요"],
            ["공부하다", "to study", "çalışmak", "공부해요"],
            ["일하다", "to work", "çalışmak (iş)", "일해요"],
            ["좋아하다", "to like", "sevmek/beğenmek", "좋아해요"],
        ],
        col_widths=[35*mm, 35*mm, 40*mm, 38*mm],
    ))

    # ── 6. Common Adjectives ──
    story += [
        sp(12), PageBreak(),
        P("6. Common Adjectives / 형용사", "h1"),
    ]
    story.append(make_table(
        ["Korean (사전형)", "English", "Türkçe", "Polite Form"],
        [
            ["크다", "big", "büyük", "커요"],
            ["작다", "small", "küçük", "작아요"],
            ["많다", "many/much", "çok", "많아요"],
            ["적다", "few/little", "az", "적어요"],
            ["맛있다", "delicious", "lezzetli", "맛있어요"],
            ["좋다", "good", "iyi", "좋아요"],
            ["나쁘다", "bad", "kötü", "나빠요"],
            ["예쁘다", "pretty", "güzel", "예뻐요"],
            ["싸다", "cheap", "ucuz", "싸요"],
            ["비싸다", "expensive", "pahalı", "비싸요"],
            ["덥다", "hot (weather)", "sıcak", "더워요"],
            ["춥다", "cold (weather)", "soğuk", "추워요"],
            ["멀다", "far", "uzak", "멀어요"],
            ["가깝다", "close/near", "yakın", "가까워요"],
        ],
        col_widths=[38*mm, 33*mm, 33*mm, 38*mm],
    ))

    # ── 7. Family Terms ──
    story += [
        sp(12),
        P("7. Family Terms / 가족", "h1"),
        P("Korean family terms vary depending on the speaker's gender and the relative's position."),
        sp(),
    ]
    story.append(make_table(
        ["Korean (한국어)", "English", "Türkçe", "Note"],
        [
            ["아버지 (아버님)", "Father", "Baba", "아버님 = formal"],
            ["어머니 (어머님)", "Mother", "Anne", "어머님 = formal"],
            ["형 / 오빠", "Older brother", "Ağabey", "형 (male) / 오빠 (female speaker)"],
            ["누나 / 언니", "Older sister", "Abla", "누나 (male) / 언니 (female speaker)"],
            ["남동생", "Younger brother", "Erkek kardeş", "Any speaker"],
            ["여동생", "Younger sister", "Kız kardeş", "Any speaker"],
            ["할아버지", "Grandfather", "Dede", "할아버님 = formal"],
            ["할머니", "Grandmother", "Büyükanne", "할머님 = formal"],
            ["아들", "Son", "Oğul", ""],
            ["딸", "Daughter", "Kız", ""],
            ["남편", "Husband", "Koca", ""],
            ["아내", "Wife", "Eş/Karı", ""],
        ],
        col_widths=[35*mm, 28*mm, 30*mm, 55*mm],
    ))

    # ── 8. Food Words ──
    story += [
        sp(12), PageBreak(),
        P("8. Food-Related Words / 음식", "h1"),
    ]
    story.append(make_table(
        ["Korean (한국어)", "English", "Türkçe"],
        [
            ["밥", "Rice / Meal", "Pilav / Yemek"],
            ["국 / 탕", "Soup / Stew", "Çorba / Güveç"],
            ["고기", "Meat", "Et"],
            ["생선", "Fish", "Balık"],
            ["채소", "Vegetables", "Sebze"],
            ["과일", "Fruit", "Meyve"],
            ["김치", "Kimchi", "Kimçi"],
            ["불고기", "Bulgogi (marinated beef)", "Bulgogi"],
            ["비빔밥", "Bibimbap (mixed rice)", "Bibimbap"],
            ["떡볶이", "Tteokbokki (spicy rice cakes)", "Tteokbokki"],
            ["라면", "Ramyeon (instant noodles)", "Ramen"],
            ["치킨", "Chicken", "Tavuk"],
            ["두부", "Tofu", "Tofu"],
            ["물", "Water", "Su"],
            ["차", "Tea", "Çay"],
            ["커피", "Coffee", "Kahve"],
            ["맥주", "Beer", "Bira"],
            ["소주", "Soju (Korean liquor)", "Soju"],
        ],
        col_widths=[50*mm, 52*mm, 52*mm],
    ))

    # ── 9. Travel Words ──
    story += [
        sp(12),
        P("9. Travel-Related Words / 여행", "h1"),
    ]
    story.append(make_table(
        ["Korean (한국어)", "English", "Türkçe"],
        [
            ["공항", "Airport", "Havalimanı"],
            ["역", "Station", "İstasyon"],
            ["버스", "Bus", "Otobüs"],
            ["지하철", "Subway", "Metro"],
            ["택시", "Taxi", "Taksi"],
            ["호텔", "Hotel", "Otel"],
            ["식당", "Restaurant", "Restoran"],
            ["병원", "Hospital", "Hastane"],
            ["약국", "Pharmacy", "Eczane"],
            ["은행", "Bank", "Banka"],
            ["화장실", "Restroom", "Tuvalet"],
            ["편의점", "Convenience store", "Market"],
            ["지도", "Map", "Harita"],
            ["여권", "Passport", "Pasaport"],
            ["표", "Ticket", "Bilet"],
        ],
        col_widths=[50*mm, 52*mm, 52*mm],
    ))

    story += [
        sp(8),
        P("Useful travel sentences / 유용한 여행 문장:", "h2"),
        P("• 여기 어떻게 가요? — How do I get here? / Buraya nasıl gidebilirim?", "example"),
        P("• 이것 얼마예요? — How much is this? / Bu ne kadar?", "example"),
        P("• 화장실이 어디에 있어요? — Where is the restroom? / Tuvalet nerede?", "example"),
        P("• 한국어를 못 해요. — I can't speak Korean. / Korece bilmiyorum.", "example"),
    ]

    doc.build(story)
    print(f"✅ Created {path}")


# ═══════════════════════════════════════════════
# PDF 3: Korean Expressions & Culture
# ═══════════════════════════════════════════════
def build_expressions_pdf(path):
    doc = SimpleDocTemplate(path, pagesize=A4,
                            topMargin=20*mm, bottomMargin=20*mm,
                            leftMargin=20*mm, rightMargin=20*mm)
    story = []

    story += [
        P("Common Korean Expressions &amp; Culture", "title"),
        P("일상 표현과 한국 문화 / Phrases, Honorifics, and Cultural Tips", "subtitle"),
        hr(), sp(10),
    ]

    # ── 1. Daily Expressions ──
    story += [
        P("1. Daily Expressions / 일상 표현", "h1"),
        sp(),
        P("1.1 Morning Routine / 아침 일과", "h2"),
    ]
    story.append(make_table(
        ["Korean (한국어)", "English", "Türkçe"],
        [
            ["좋은 아침이에요!", "Good morning!", "Günaydın!"],
            ["잘 자셨어요?", "Did you sleep well?", "İyi uyudunuz mu?"],
            ["오늘 날씨가 좋네요.", "The weather is nice today.", "Bugün hava güzel."],
            ["아침 먹었어요?", "Did you eat breakfast?", "Kahvaltı yaptın mı?"],
            ["오늘 바빠요?", "Are you busy today?", "Bugün meşgul müsün?"],
        ],
        col_widths=[48*mm, 52*mm, 55*mm],
    ))

    story += [
        sp(),
        P("1.2 At a Restaurant / 식당에서", "h2"),
    ]
    story.append(make_table(
        ["Korean (한국어)", "English", "Türkçe"],
        [
            ["메뉴판 주세요.", "Please give me the menu.", "Menüyü verir misiniz?"],
            ["이것 주세요.", "Please give me this.", "Bunu verin lütfen."],
            ["맛있게 드세요!", "Enjoy your meal!", "Afiyet olsun!"],
            ["계산서 주세요.", "Check please.", "Hesap lütfen."],
            ["너무 맛있어요!", "It's very delicious!", "Çok lezzetli!"],
            ["배불러요.", "I'm full.", "Doydum."],
            ["매워요!", "It's spicy!", "Acı!"],
            ["더 주세요.", "Please give me more.", "Daha verin lütfen."],
        ],
        col_widths=[48*mm, 48*mm, 55*mm],
    ))

    story += [
        sp(),
        P("1.3 Shopping / 쇼핑", "h2"),
    ]
    story.append(make_table(
        ["Korean (한국어)", "English", "Türkçe"],
        [
            ["얼마예요?", "How much is it?", "Ne kadar?"],
            ["너무 비싸요.", "It's too expensive.", "Çok pahalı."],
            ["깎아 주세요.", "Please give a discount.", "İndirim yapar mısınız?"],
            ["이거 살게요.", "I'll buy this.", "Bunu alacağım."],
            ["카드로 될까요?", "Can I pay by card?", "Kartla olur mu?"],
            ["영수증 주세요.", "Receipt please.", "Fiş lütfen."],
        ],
        col_widths=[48*mm, 48*mm, 55*mm],
    ))

    story += [
        sp(),
        P("1.4 Asking for Directions / 길 묻기", "h2"),
    ]
    story.append(make_table(
        ["Korean (한국어)", "English", "Türkçe"],
        [
            ["여기가 어디예요?", "Where is this?", "Burası neresi?"],
            ["지하철역이 어디에 있어요?", "Where is the subway?", "Metro istasyonu nerede?"],
            ["오른쪽으로 가세요.", "Go to the right.", "Sağa gidin."],
            ["왼쪽으로 가세요.", "Go to the left.", "Sola gidin."],
            ["직진하세요.", "Go straight.", "Düz gidin."],
            ["얼마나 걸려요?", "How long does it take?", "Ne kadar sürer?"],
        ],
        col_widths=[48*mm, 48*mm, 55*mm],
    ))

    # ── 2. Polite Phrases ──
    story += [
        sp(12), PageBreak(),
        P("2. Polite Phrases / 존댓말 표현", "h1"),
        P("Politeness is central to Korean communication. Using the right level of "
          "formality shows respect and social awareness."),
        sp(),
    ]
    story.append(make_table(
        ["Korean (한국어)", "English", "Türkçe", "When to Use"],
        [
            ["수고하세요", "Good job / Take care", "Kolay gelsin", "Leaving work"],
            ["실례합니다", "Excuse me", "Affedersiniz", "Getting attention"],
            ["괜찮아요", "It's okay", "Sorun değil", "Accepting apology"],
            ["천만에요", "You're welcome", "Rica ederim", "After thanks"],
            ["잘 부탁드립니다", "Please take care of it", "Size güveniyorum", "Requesting help"],
            ["신세 지세요", "Take care (health)", "Kendinize dikkat edin", "Caring goodbye"],
            ["축하합니다!", "Congratulations!", "Tebrikler!", "Celebrations"],
            ["힘내세요!", "Cheer up!", "Gayret!", "Encouraging"],
            ["죄송하지만...", "I'm sorry but...", "Özür dilerim ama...", "Polite refusal"],
            ["다음에 또 봐요", "See you next time", "Sonra görüşürüz", "Casual goodbye"],
        ],
        col_widths=[37*mm, 37*mm, 37*mm, 38*mm],
    ))

    # ── 3. Honorific System ──
    story += [
        sp(12),
        P("3. Korean Honorific System / 높임말", "h1"),
        P("The Korean honorific system (높임말) is one of the most complex "
          "and important aspects of the language. It reflects Korean society's deep "
          "emphasis on respect, age, and social hierarchy."),
        sp(),

        P("3.1 Why Honorifics Matter / 높임말이 중요한 이유", "h2"),
        P("In Korean culture, the relationship between speaker and listener determines "
          "the language used. Using incorrect speech levels can be seen as rude or "
          "awkward. Age, social status, and familiarity all play a role."),
        sp(),

        P("3.2 Subject Honorific: -(으)시다", "h2"),
        P("When the <b>subject</b> of the sentence is someone you want to show respect to "
          "(elders, teachers, bosses), add -(으)시다 to the verb stem."),
        sp(),
    ]
    story.append(make_table(
        ["Regular Verb", "Honorific Verb", "Meaning"],
        [
            ["먹다", "드시다 (잡수시다)", "to eat (honorific)"],
            ["자다", "주무시다", "to sleep (honorific)"],
            ["있다", "계시다", "to exist/be (honorific)"],
            ["말하다", "말씀하시다", "to speak (honorific)"],
            ["죽다", "돌아가시다", "to pass away (honorific)"],
            ["보다", "보시다", "to see (honorific)"],
        ],
        col_widths=[45*mm, 55*mm, 55*mm],
    ))

    story += [
        sp(),
        P("Examples / 예문:", "h2"),
        P("• 할머니가 진지를 <b>드셨어요</b>. — Grandmother ate. (honorific for 먹었어요)", "bullet"),
        P("• 선생님이 <b>가르치셨어요</b>. — The teacher taught. (honorific)", "bullet"),
        P("• 아버지가 <b>주무세요</b>. — Father is sleeping. (honorific)", "bullet"),
        sp(),

        P("3.3 Honorific Nouns / 높임 명사", "h2"),
        P("Some nouns have special honorific forms:"),
        sp(),
    ]
    story.append(make_table(
        ["Regular (일반)", "Honorific (높임)", "English"],
        [
            ["이름", "성함", "Name"],
            ["나이", "연세", "Age"],
            ["집", "댁", "House/Home"],
            ["말", "말씀", "Words/Speech"],
            ["밥", "진지", "Meal/Rice"],
            ["생일", "생신", "Birthday"],
            ["아프다", "편찮으시다", "To be sick/unwell"],
        ],
        col_widths=[50*mm, 52*mm, 52*mm],
    ))

    story += [
        sp(),
        P("3.4 Honorific Particles / 높임 조사", "h2"),
        P("The subject particle 이/가 becomes <b>께서</b> when referring to someone you respect."),
        P("• 선생님<b>께서</b> 오셨어요. — The teacher came. (honorific)", "bullet"),
        P("• 사장님<b>께서</b> 말씀하셨어요. — The boss spoke. (honorific)", "bullet"),
        sp(),

        P("3.5 Address Terms and Titles / 호칭", "h2"),
        P("Koreans rarely use names alone. Titles and relationship terms are used extensively:"),
        sp(),
        P("• 선생님 — Teacher (also for doctors, lawyers, respected professionals)", "bullet"),
        P("• 사장님 — Company president / Boss", "bullet"),
        P("• 부장님 — Department head", "bullet"),
        P("• 아저씨 — Mister (to strangers, slightly informal)", "bullet"),
        P("• 아줌마 — Ma'am / Aunt (to older women, friendly)", "bullet"),
        P("• [Name]+씨 — Mr./Ms. [Name] (polite but not very formal)", "bullet"),
        sp(12),
    ]

    # ── 4. Cultural Tips ──
    story += [
        PageBreak(),
        P("4. Cultural Tips for Korean Learners / 한국 문화 팁", "h1"),
        sp(),

        P("4.1 Age and Hierarchy / 나이와 서열", "h2"),
        P("Age is perhaps the most important factor in Korean social interactions. "
          "When Koreans meet for the first time, they often ask each other's age to "
          "determine the appropriate speech level. Even a one-year difference can change "
          "the dynamic of a conversation."),
        P("<b>Tip:</b> When meeting Koreans, don't be surprised if they ask your age early "
          "in the conversation. It's not rude — it helps them determine how to address you.", "note"),
        sp(),

        P("4.2 Bowing / 인사 (절)", "h2"),
        P("Bowing is the traditional Korean greeting. The depth and duration depends on "
          "the relative status:"),
        P("• Slight nod (15°): casual greeting between equals", "bullet"),
        P("• Standard bow (30°): greeting elders, business settings", "bullet"),
        P("• Deep bow (45°+): showing deep respect, formal apologies", "bullet"),
        sp(),

        P("4.3 Dining Etiquette / 식사 예절", "h2"),
        P("• Wait for the eldest person to start eating before you begin.", "bullet"),
        P("• Don't pour your own drink — others pour for you, and you pour for them.", "bullet"),
        P("• Hold your glass with both hands when receiving a drink from an elder.", "bullet"),
        P("• Turn your head slightly away when drinking in front of an elder.", "bullet"),
        P("• Say 잘 먹겠습니다 (I will eat well) before a meal.", "bullet"),
        P("• Say 잘 먹었습니다 (I ate well) after a meal.", "bullet"),
        sp(),

        P("4.4 Removing Shoes / 신발 벗기", "h2"),
        P("Always remove your shoes when entering a Korean home, and often in traditional "
          "restaurants. Look for a shoe rack or follow what others do."),
        sp(),

        P("4.5 Two Hands Rule / 두 손 규칙", "h2"),
        P("When giving or receiving objects (especially from elders or in formal settings), "
          "use both hands or support your right arm with your left hand. This applies to "
          "business cards, gifts, money, and drinks."),
        sp(),

        P("4.6 Korean Age System / 한국 나이", "h2"),
        P("Korea traditionally uses a different age counting system. In 2023, South Korea "
          "officially adopted the international age system for legal purposes. However, "
          "the traditional system (한국 나이) is still widely used:"),
        P("• You are 1 year old at birth.", "bullet"),
        P("• Everyone gains one year on January 1st, not on their birthday.", "bullet"),
        P("<b>Tip:</b> If a Korean tells you their age, ask whether they mean "
          "한국 나이 (Korean age) or 만 나이 (international age).", "note"),
        sp(),

        P("4.7 Useful Cultural Expressions / 문화 표현", "h2"),
    ]
    story.append(make_table(
        ["Korean (한국어)", "Meaning", "Cultural Context"],
        [
            ["화이팅!", "Fighting! (You can do it!)", "Encouragement cheer"],
            ["눈치", "Nunchi (social awareness)", "Reading the room"],
            ["정", "Jeong (deep affection/bond)", "Untranslatable emotional bond"],
            ["한", "Han (deep sorrow/resentment)", "Cultural emotion concept"],
            ["어버이날", "Parents' Day (May 8)", "National holiday"],
            ["추석", "Chuseok (harvest festival)", "Korean Thanksgiving"],
            ["설날", "Seollal (Lunar New Year)", "Major family holiday"],
            ["노래방", "Noraebang (karaoke room)", "Popular social activity"],
            ["찜질방", "Jjimjilbang (spa/bathhouse)", "Korean bathhouse culture"],
        ],
        col_widths=[35*mm, 45*mm, 70*mm],
    ))

    story += [
        sp(10),
        P("<b>Final Tip:</b> The best way to learn Korean is to immerse yourself in the "
          "culture. Watch Korean dramas (드라마), listen to K-pop, and practice with native "
          "speakers. Don't be afraid of making mistakes — Koreans appreciate the effort!", "note"),
    ]

    doc.build(story)
    print(f"✅ Created {path}")


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)

    build_grammar_pdf(os.path.join(data_dir, "korean_grammar.pdf"))
    build_vocabulary_pdf(os.path.join(data_dir, "korean_vocabulary.pdf"))
    build_expressions_pdf(os.path.join(data_dir, "korean_expressions.pdf"))
    print("\n🎉 All PDFs generated successfully!")
