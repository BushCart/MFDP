from src.chunking.simple_splitter import text_splitter


def test_returns_list_of_dicts():
    text = "Катя побежала. За спиной ее грохнул залп. Одна из пуль просвистела возле виска, другая вырвала клок из рукава полушубка."
    result = text_splitter(text, max_length=50, overlap=10)
    assert isinstance(result, list)
    assert all(isinstance(chunk, dict) for chunk in result)
    assert all("text" in chunk and "chunk_id" in chunk for chunk in result)


def test_split_ends_on_punctuation():
    text = "У домика одна стена короткая, одна подлиннее. Зовет в гости, поиграть в саду!"\
    "В беседке, увитой виноградом, сидит серый человек — папа. В папе сидит рак."
    result = text_splitter(text, max_length=20, overlap=0)
    assert "." in result[0]["text"] or "!" in result[0]["text"], "Чанк не обрезан по знаку препинания"


def test_no_negative_overlap():
    text = "Hello World. " * 20
    result = text_splitter(text, max_length=50, overlap=10)
    for chunk in result:
        assert len(chunk["text"]) > 0


def test_short_text_returns_one_chunk():
    text = "Короткий текст."
    result = text_splitter(text, max_length=100)
    assert len(result) == 1
    assert result[0]["text"] == text


def test_empty_text_returns_empty_list():
    text = ""
    result = text_splitter(text)
    assert result == []

def test_overlap_is_respected():
    text = "В 1800-х годах, в те времена, когда не было еще ни железных, ни шоссейных дорог," \
    " ни газового, ни стеаринового света, ни пружинных низких диванов, ни мебели без лаку," \
    " ни разочарованных юношей со стеклышками, ни либеральных философов-женщин, ни милых дам-камелий," \
    " которых так много развелось в наше время, - в те наивные времена, когда из Москвы," \
    " выезжая в Петербург в повозке или карете, брали с собой целую кухню домашнего приготовления," \
    " ехали восемь суток по мягкой, пыльной или грязной дороге и верили в пожарские котлеты," \
    " в валдайские колокольчики и бублики, - когда в длинные осенние вечера нагорали сальные свечи," \
    " освещая семейные кружки из двадцати и тридцати человек, на балах в канделябры вставлялись восковые и спермацетовые свечи," \
    " когда мебель ставили симметрично, когда наши отцы были еще молоды не одним отсутствием морщин и седых волос," \
    " а стрелялись за женщин и из другого угла комнаты бросались поднимать нечаянно и не нечаянно уроненные платочки," \
    " наши матери носили коротенькие талии и огромные рукава и решали семейные дела выниманием билетиков," \
    " когда прелестные дамы-камелии прятались от дневного света, - в наивные времена масонских лож," \
    " мартинистов, тугендбунда, во времена Милорадовичей, Давыдовых, Пушкиных, - в губернском городе К." \
    " был съезд помещиков, и кончались дворянские выборы."
    result = text_splitter(text, max_length=50, overlap=10)
    for i in range(len(result)-1):
        prev = result[i]["text"]
        curr = result[i+1]["text"]
        if len(prev) >= 20 and len(curr) >= 20:
            assert prev[-10:] == curr[:10], f"Чанки {i} и {i+1} не перекрываются как ожидалось"