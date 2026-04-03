"""
memweave/search/keyword.py — FTS5 keyword search using SQLite BM25.

The search itself uses SQLite FTS5's native ``bm25()`` function and
``MATCH`` operator — no external dependencies.
"""

from __future__ import annotations

import re

import aiosqlite

from memweave.search.strategy import RawSearchRow

# ── Stop word sets ────────────────────────────────────────────────────────────
# All lowercase; matched against lowercased tokens.

_STOP_WORDS_EN: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "this",
        "that",
        "these",
        "those",
        "i",
        "me",
        "my",
        "we",
        "our",
        "you",
        "your",
        "he",
        "she",
        "it",
        "they",
        "them",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "can",
        "may",
        "might",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "about",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "under",
        "over",
        "and",
        "or",
        "but",
        "if",
        "then",
        "because",
        "as",
        "while",
        "when",
        "where",
        "what",
        "which",
        "who",
        "how",
        "why",
        "yesterday",
        "today",
        "tomorrow",
        "earlier",
        "later",
        "recently",
        "ago",
        "just",
        "now",
        "thing",
        "things",
        "stuff",
        "something",
        "anything",
        "everything",
        "nothing",
        "please",
        "help",
        "find",
        "show",
        "get",
        "tell",
        "give",
    }
)

_STOP_WORDS_ES: frozenset[str] = frozenset(
    {
        "el",
        "la",
        "los",
        "las",
        "un",
        "una",
        "unos",
        "unas",
        "este",
        "esta",
        "ese",
        "esa",
        "yo",
        "me",
        "mi",
        "nosotros",
        "nosotras",
        "tu",
        "tus",
        "usted",
        "ustedes",
        "ellos",
        "ellas",
        "de",
        "del",
        "a",
        "en",
        "con",
        "por",
        "para",
        "sobre",
        "entre",
        "y",
        "o",
        "pero",
        "si",
        "porque",
        "como",
        "es",
        "son",
        "fue",
        "fueron",
        "ser",
        "estar",
        "haber",
        "tener",
        "hacer",
        "ayer",
        "hoy",
        "mañana",
        "antes",
        "despues",
        "después",
        "ahora",
        "recientemente",
        "que",
        "qué",
        "cómo",
        "cuando",
        "cuándo",
        "donde",
        "dónde",
        "porquê",
        "favor",
        "ayuda",
    }
)

_STOP_WORDS_PT: frozenset[str] = frozenset(
    {
        "o",
        "a",
        "os",
        "as",
        "um",
        "uma",
        "uns",
        "umas",
        "este",
        "esta",
        "esse",
        "essa",
        "eu",
        "me",
        "meu",
        "minha",
        "nos",
        "nós",
        "você",
        "vocês",
        "ele",
        "ela",
        "eles",
        "elas",
        "de",
        "do",
        "da",
        "em",
        "com",
        "por",
        "para",
        "sobre",
        "entre",
        "e",
        "ou",
        "mas",
        "se",
        "porque",
        "como",
        "é",
        "são",
        "foi",
        "foram",
        "ser",
        "estar",
        "ter",
        "fazer",
        "ontem",
        "hoje",
        "amanhã",
        "antes",
        "depois",
        "agora",
        "recentemente",
        "que",
        "quê",
        "quando",
        "onde",
        "porquê",
        "favor",
        "ajuda",
    }
)

_STOP_WORDS_AR: frozenset[str] = frozenset(
    {
        "ال",
        "و",
        "أو",
        "لكن",
        "ثم",
        "بل",
        "أنا",
        "نحن",
        "هو",
        "هي",
        "هم",
        "هذا",
        "هذه",
        "ذلك",
        "تلك",
        "هنا",
        "هناك",
        "من",
        "إلى",
        "الى",
        "في",
        "على",
        "عن",
        "مع",
        "بين",
        "ل",
        "ب",
        "ك",
        "كان",
        "كانت",
        "يكون",
        "تكون",
        "صار",
        "أصبح",
        "يمكن",
        "ممكن",
        "بالأمس",
        "امس",
        "اليوم",
        "غدا",
        "الآن",
        "قبل",
        "بعد",
        "مؤخرا",
        "لماذا",
        "كيف",
        "ماذا",
        "متى",
        "أين",
        "هل",
        "من فضلك",
        "فضلا",
        "ساعد",
    }
)

_STOP_WORDS_KO: frozenset[str] = frozenset(
    {
        "은",
        "는",
        "이",
        "가",
        "을",
        "를",
        "의",
        "에",
        "에서",
        "로",
        "으로",
        "와",
        "과",
        "도",
        "만",
        "까지",
        "부터",
        "한테",
        "에게",
        "께",
        "처럼",
        "같이",
        "보다",
        "마다",
        "밖에",
        "대로",
        "나",
        "나는",
        "내가",
        "나를",
        "너",
        "우리",
        "저",
        "저희",
        "그",
        "그녀",
        "그들",
        "이것",
        "저것",
        "그것",
        "여기",
        "저기",
        "거기",
        "있다",
        "없다",
        "하다",
        "되다",
        "이다",
        "아니다",
        "보다",
        "주다",
        "오다",
        "가다",
        "것",
        "거",
        "등",
        "수",
        "때",
        "곳",
        "중",
        "분",
        "잘",
        "더",
        "또",
        "매우",
        "정말",
        "아주",
        "많이",
        "너무",
        "좀",
        "그리고",
        "하지만",
        "그래서",
        "그런데",
        "그러나",
        "또는",
        "그러면",
        "왜",
        "어떻게",
        "뭐",
        "언제",
        "어디",
        "누구",
        "무엇",
        "어떤",
        "어제",
        "오늘",
        "내일",
        "최근",
        "지금",
        "아까",
        "나중",
        "전에",
        "제발",
        "부탁",
    }
)

_STOP_WORDS_JA: frozenset[str] = frozenset(
    {
        "これ",
        "それ",
        "あれ",
        "この",
        "その",
        "あの",
        "ここ",
        "そこ",
        "あそこ",
        "する",
        "した",
        "して",
        "です",
        "ます",
        "いる",
        "ある",
        "なる",
        "できる",
        "の",
        "こと",
        "もの",
        "ため",
        "そして",
        "しかし",
        "また",
        "でも",
        "から",
        "まで",
        "より",
        "だけ",
        "なぜ",
        "どう",
        "何",
        "いつ",
        "どこ",
        "誰",
        "どれ",
        "昨日",
        "今日",
        "明日",
        "最近",
        "今",
        "さっき",
        "前",
        "後",
    }
)

_STOP_WORDS_ZH: frozenset[str] = frozenset(
    {
        "我",
        "我们",
        "你",
        "你们",
        "他",
        "她",
        "它",
        "他们",
        "这",
        "那",
        "这个",
        "那个",
        "这些",
        "那些",
        "的",
        "了",
        "着",
        "过",
        "得",
        "地",
        "吗",
        "呢",
        "吧",
        "啊",
        "呀",
        "嘛",
        "啦",
        "是",
        "有",
        "在",
        "被",
        "把",
        "给",
        "让",
        "用",
        "到",
        "去",
        "来",
        "做",
        "说",
        "看",
        "找",
        "想",
        "要",
        "能",
        "会",
        "可以",
        "和",
        "与",
        "或",
        "但",
        "但是",
        "因为",
        "所以",
        "如果",
        "虽然",
        "而",
        "也",
        "都",
        "就",
        "还",
        "又",
        "再",
        "才",
        "只",
        "之前",
        "以前",
        "之后",
        "以后",
        "刚才",
        "现在",
        "昨天",
        "今天",
        "明天",
        "最近",
        "东西",
        "事情",
        "事",
        "什么",
        "哪个",
        "哪些",
        "怎么",
        "为什么",
        "多少",
        "请",
        "帮",
        "帮忙",
        "告诉",
    }
)

# Unicode-aware token pattern — matches letters, digits, underscore
# (equivalent to /[\p{L}\p{N}_]+/gu)
_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


# ── Public helpers ────────────────────────────────────────────────────────────


def build_fts_query(raw: str) -> str | None:
    """Convert a raw user query string into a safe FTS5 MATCH expression.

    Algorithm:
    1. Tokenize ``raw`` with a Unicode-aware word regex (letters, digits,
       underscores).  Hyphens, spaces, and punctuation act as separators.
    2. Strip any embedded double-quote characters from each token (FTS5
       uses ``"..."`` for phrase queries — leaving quotes in would break the
       MATCH expression).
    3. Wrap each token in double quotes.
    4. Join with `` AND `` so all tokens must appear in matching chunks.
    5. Return ``None`` if the query yields no tokens (e.g. pure punctuation).

    Args:
        raw: Arbitrary user input, e.g. ``"which database did we pick?"``.

    Returns:
        FTS5 MATCH string, e.g. ``'"which" AND "database" AND "did" AND "we" AND "pick"'``,
        or ``None`` if no tokens were found.

    Examples::

        build_fts_query("hello world")
        # → '"hello" AND "world"'

        build_fts_query("FOO_bar baz-1")
        # → '"FOO_bar" AND "baz" AND "1"'

        build_fts_query("???")
        # → None
    """
    tokens = [t.strip() for t in _TOKEN_RE.findall(raw) if t.strip()]
    if not tokens:
        return None
    # Strip double quotes to avoid breaking FTS5 phrase syntax
    quoted = [f'"{t.replace(chr(34), "")}"' for t in tokens]
    return " AND ".join(quoted)


def bm25_rank_to_score(rank: float) -> float:
    """Convert an FTS5 BM25 rank to a normalized relevance score in [0, 1].

    FTS5's ``bm25()`` function returns **negative** values for matches — a more
    negative rank means a *better* match (higher relevance).  Zero or positive
    values indicate no match or very weak relevance.

    Conversion formula:
    - Non-finite (NaN / ±Inf): return near-zero fallback ``1 / (1 + 999)``.
    - rank < 0 (relevant match):  ``(-rank) / (1 + (-rank))``
      → monotonically increases toward 1 as relevance grows.
    - rank ≥ 0 (no match / weak): ``1 / (1 + rank)``
      → monotonically decreases toward 0.

    Args:
        rank: Raw ``bm25(table)`` value from SQLite.

    Returns:
        Normalized score in (0, 1].

    Examples::

        bm25_rank_to_score(-4.2)   # → ~0.808  (strong match)
        bm25_rank_to_score(-0.5)   # → ~0.333  (weak match)
        bm25_rank_to_score(0.0)    # → 1.0     (neutral — returns 1/(1+0))
        bm25_rank_to_score(float("nan"))  # → ~0.001
    """
    import math

    if not math.isfinite(rank):
        return 1.0 / (1.0 + 999.0)
    if rank < 0:
        relevance = -rank
        return relevance / (1.0 + relevance)
    return 1.0 / (1.0 + rank)


def is_stop_word(token: str) -> bool:
    """Return True if ``token`` appears in any of the seven language stop word sets.
    Comparison is case-sensitive and matches the stored forms directly (English
    tokens should be lowercased before calling).

    Args:
        token: A single word token.

    Returns:
        ``True`` if the token is a stop word in EN, ES, PT, AR, ZH, KO, or JA.
    """
    return (
        token in _STOP_WORDS_EN
        or token in _STOP_WORDS_ES
        or token in _STOP_WORDS_PT
        or token in _STOP_WORDS_AR
        or token in _STOP_WORDS_ZH
        or token in _STOP_WORDS_KO
        or token in _STOP_WORDS_JA
    )


def _is_valid_keyword(token: str) -> bool:
    """Return True if ``token`` is a meaningful search keyword.

    Filters:
    - Empty strings.
    - Pure ASCII alpha tokens shorter than 3 characters (likely fragments).
    - Pure digit strings (not useful for semantic search).
    - Tokens made entirely of punctuation / symbols.

    Args:
        token: A candidate keyword token.

    Returns:
        ``True`` if the token should be kept as a keyword.
    """
    if not token:
        return False
    # Very short pure-ASCII words (fragments / likely stop words)
    if re.match(r"^[a-zA-Z]+$", token) and len(token) < 3:
        return False
    # Pure numbers
    if re.match(r"^\d+$", token):
        return False
    # All punctuation / symbols — use \W (non-word chars) as an approximation;
    # our tokenizer already strips non-word chars, so this guard is mainly for
    # tokens that somehow consist entirely of underscores or similar edge cases.
    if re.match(r"^\W+$", token, re.UNICODE):
        return False
    return True


def extract_keywords(query: str) -> list[str]:
    """Extract meaningful keywords from a search query.

    Steps:
    1. Tokenize ``query`` using the Unicode word regex.
    2. Lowercase each token for stop word comparison.
    3. Skip stop words (all seven language sets).
    4. Skip invalid keywords (short ASCII, pure numbers, punctuation).
    5. Deduplicate while preserving order.

    This is used by the query-expansion path (FTS-only fallback) to build a
    broader OR query from meaningful terms.

    Args:
        query: Raw user query string.

    Returns:
        Ordered list of unique, meaningful keyword tokens.

    Examples::

        extract_keywords("which database did we pick?")
        # → ["database", "pick"]  ("which", "did", "we" are stop words)

        extract_keywords("PostgreSQL connection pooling")
        # → ["PostgreSQL", "connection", "pooling"]
    """
    tokens = _TOKEN_RE.findall(query)
    keywords: list[str] = []
    seen: set[str] = set()
    for raw_token in tokens:
        token_lower = raw_token.lower()
        if is_stop_word(token_lower) or is_stop_word(raw_token):
            continue
        if not _is_valid_keyword(raw_token):
            continue
        if token_lower in seen:
            continue
        seen.add(token_lower)
        keywords.append(raw_token)
    return keywords


# ── KeywordSearch backend ─────────────────────────────────────────────────────


class KeywordSearch:
    """FTS5 BM25 keyword search backend.

    Uses SQLite's built-in FTS5 extension with the ``bm25()`` ranking function.
    No external dependencies required.

    The ``chunks_fts`` virtual table is created by ``ensure_schema()`` in
    ``memweave/storage/schema.py``.  Its schema::

        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            text,
            id UNINDEXED,
            path UNINDEXED,
            source UNINDEXED,
            model UNINDEXED,
            start_line UNINDEXED,
            end_line UNINDEXED
        );

    Usage::

        ks = KeywordSearch()
        rows = await ks.search(db, "PostgreSQL connection", None, model, limit=10)
    """

    async def search(
        self,
        db: aiosqlite.Connection,
        query: str,
        query_vec: list[float] | None,  # noqa: ARG002 — unused, satisfies protocol
        model: str,
        limit: int,
        *,
        source_filter: str | None = None,
    ) -> list[RawSearchRow]:
        """Run an FTS5 BM25 keyword search.

        Steps:
        1. Build a safe FTS5 MATCH expression from ``query`` via
           :func:`build_fts_query`.  Returns empty list if query yields no
           tokens.
        2. Execute::

               SELECT id, path, source, start_line, end_line, text,
                      bm25(chunks_fts) AS rank
                 FROM chunks_fts
                WHERE chunks_fts MATCH ?
                  AND model = ?
                  [AND source = ?]
                ORDER BY rank ASC
                LIMIT ?

           ``ORDER BY rank ASC`` because FTS5 BM25 returns *negative* values
           for relevant matches — the most negative rank is the best match.
        3. Convert each ``rank`` value to a [0, 1] score via
           :func:`bm25_rank_to_score`.

        Args:
            db:            Open aiosqlite connection.
            query:         Raw user query.
            query_vec:     Ignored (keyword search doesn't use embeddings).
            model:         Embedding model name to filter chunks.
            limit:         Maximum rows to return.
            source_filter: ``"memory"`` or ``"sessions"`` to restrict results.

        Returns:
            List of :class:`~memweave.search.strategy.RawSearchRow`, ordered by
            descending relevance.  Empty list when query has no tokens or no
            rows match.
        """
        fts_query = build_fts_query(query)
        if fts_query is None:
            return []

        source_clause = " AND source = ?" if source_filter else ""
        params: list[object] = [fts_query, model]
        if source_filter:
            params.append(source_filter)
        params.append(limit)

        sql = (
            "SELECT id, path, source, start_line, end_line, text,"
            "       bm25(chunks_fts) AS rank"
            "  FROM chunks_fts"
            " WHERE chunks_fts MATCH ?"
            "   AND model = ?"
            f"{source_clause}"
            " ORDER BY rank ASC"
            " LIMIT ?"
        )

        rows: list[RawSearchRow] = []
        async with db.execute(sql, params) as cursor:
            async for row in cursor:
                chunk_id, path, source, start_line, end_line, text, rank = row
                score = bm25_rank_to_score(float(rank))
                rows.append(
                    RawSearchRow(
                        chunk_id=chunk_id,
                        path=path,
                        source=source,
                        start_line=start_line,
                        end_line=end_line,
                        text=text,
                        score=score,
                        text_score=score,
                    )
                )
        return rows
