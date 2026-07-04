# -*- coding: utf-8 -*-
"""
(주)한빛파트너스 HR 지식봇 — Streamlit 웹 클라이언트
Amazon Bedrock Knowledge Base(retrieve_and_generate)와 연동한 채팅 앱

실행:  streamlit run hr_chatbot_app.py
필요:  pip install streamlit boto3  (AWS 자격 증명 설정 필요)
"""
import boto3
import streamlit as st

# ─────────────────────────── 기본 설정 ───────────────────────────
st.set_page_config(page_title='한빛파트너스 HR 지식봇', page_icon='🏢', layout='centered')

REGION = 'us-east-1'

# {account}는 실행 시 실제 계정 ID로 치환됩니다 (foundation-model ARN은 계정 없이 사용)
MODEL_OPTIONS = {
    'Claude Sonnet 4.6 (권장)': f'arn:aws:bedrock:{{account_region}}:{{account}}:inference-profile/us.anthropic.claude-sonnet-4-6',
    'Claude 3 Haiku (빠름·저렴)': f'arn:aws:bedrock:{{account_region}}::foundation-model/anthropic.claude-3-haiku-20240307-v1:0',
}

PROMPT_TEMPLATE = """
당신은 (주)한빛파트너스의 사내 규정 안내 AI입니다. 아래 검색된 규정 문서를 근거로 정확하게 답변하세요.
- 검색된 규정에 없는 내용은 '규정에서 확인되지 않습니다. 담당 부서에 문의해 주세요.'라고 안내하세요.
- 수치(일수, 금액, 기한)는 규정 원문 그대로 인용하세요.
- 답변은 간결하게, 공손한 어투로 작성하세요.
\n\n$search_results$\n\n질문: $query$
"""


@st.cache_resource
def get_clients():
    """boto3 클라이언트와 계정 ID (앱 전체에서 1회만 생성)"""
    runtime = boto3.client('bedrock-agent-runtime', region_name=REGION)
    account = boto3.client('sts').get_caller_identity()['Account']
    return runtime, account


def rag_query(question, kb_id, model_arn, session_id=None, top_k=5):
    """Knowledge Base 검색 + 답변 생성 (세션 유지)"""
    runtime, _ = get_clients()
    kwargs = {
        'input': {'text': question},
        'retrieveAndGenerateConfiguration': {
            'type': 'KNOWLEDGE_BASE',
            'knowledgeBaseConfiguration': {
                'knowledgeBaseId': kb_id,
                'modelArn': model_arn,
                'retrievalConfiguration': {
                    'vectorSearchConfiguration': {'numberOfResults': top_k}
                },
                'generationConfiguration': {
                    'promptTemplate': {'textPromptTemplate': PROMPT_TEMPLATE}
                },
            },
        },
    }
    if session_id:
        kwargs['sessionId'] = session_id

    resp = runtime.retrieve_and_generate(**kwargs)
    answer = resp['output']['text']

    # 인용에서 (파일명, 원문 일부) 추출
    sources = []
    for c in resp.get('citations', []):
        for ref in c.get('retrievedReferences', []):
            uri = ref.get('location', {}).get('s3Location', {}).get('uri', '')
            name = uri.split('/')[-1]
            snippet = ref.get('content', {}).get('text', '')[:200]
            if name and name not in [s[0] for s in sources]:
                sources.append((name, snippet))
    return answer, sources, resp.get('sessionId')


# ─────────────────────────── 사이드바 ───────────────────────────
with st.sidebar:
    st.header('⚙️ 설정')
    kb_id = st.text_input('Knowledge Base ID', value='', placeholder='예: ABCDE12345')
    model_label = st.selectbox('생성 모델', list(MODEL_OPTIONS.keys()))
    top_k = st.slider('검색 청크 수 (numberOfResults)', 1, 10, 5)
    st.divider()
    if st.button('🔄 새 대화 시작', use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = None
        st.rerun()
    st.caption('세션이 유지되는 동안 이전 질문의 맥락을 기억합니다.')

# ─────────────────────────── 메인 화면 ───────────────────────────
st.title('🏢 한빛파트너스 HR 지식봇')
st.caption('사내 규정 7종을 근거로 답하는 RAG 챗봇 · Amazon Bedrock Knowledge Base')

if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'session_id' not in st.session_state:
    st.session_state.session_id = None

# 예시 질문 버튼 (첫 화면에만)
if not st.session_state.messages:
    st.markdown('##### 이런 걸 물어보세요')
    examples = ['연차휴가는 기본 며칠인가요?', '출장 숙박비 한도는 얼마인가요?', '노트북이 고장나면 어디에 문의하나요?']
    cols = st.columns(len(examples))
    for col, ex in zip(cols, examples):
        if col.button(ex, use_container_width=True):
            st.session_state.pending = ex
            st.rerun()

# 대화 이력 렌더링
for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])
        if msg.get('sources'):
            with st.expander(f"📎 근거 규정 {len(msg['sources'])}건"):
                for name, snippet in msg['sources']:
                    st.markdown(f'**{name}**')
                    st.caption(f'"{snippet}..."')

# 입력 처리 (채팅 입력창 또는 예시 버튼)
question = st.chat_input('규정에 대해 질문하세요...')
if not question and 'pending' in st.session_state:
    question = st.session_state.pop('pending')

if question:
    if not kb_id:
        st.error('왼쪽 사이드바에 Knowledge Base ID를 먼저 입력하세요.')
        st.stop()

    st.session_state.messages.append({'role': 'user', 'content': question})
    with st.chat_message('user'):
        st.markdown(question)

    with st.chat_message('assistant'):
        with st.spinner('규정을 검색하고 있습니다...'):
            try:
                _, account = get_clients()
                model_arn = MODEL_OPTIONS[model_label].format(account_region=REGION, account=account)
                answer, sources, st.session_state.session_id = rag_query(
                    question, kb_id, model_arn,
                    st.session_state.session_id, top_k)
            except Exception as e:
                st.error(f'오류가 발생했습니다: {e}')
                st.stop()
        st.markdown(answer)
        if sources:
            with st.expander(f'📎 근거 규정 {len(sources)}건'):
                for name, snippet in sources:
                    st.markdown(f'**{name}**')
                    st.caption(f'"{snippet}..."')

    st.session_state.messages.append({'role': 'assistant', 'content': answer, 'sources': sources})
