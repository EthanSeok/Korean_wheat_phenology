import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import tempfile
import os
from thermal_time import APSIMWheatPhenology
from wheat_stage import wheat_stage_process

st.set_page_config(
    page_title="APSIM Wheat Phenology Model",
    page_icon="🌾",
    layout="wide"
)

# 한국 밀 품종별 최적화 파라미터 (Table 3 기반)
KOREAN_WHEAT_CULTIVARS = {
    "조광": {
        "photop_sens": 2.6,
        "vern_sens": 3.8,
        "tt_start_grain_fill": 660,
        "description": "Jogwang - 조광"
    },
    "우리": {
        "photop_sens": 2.3,
        "vern_sens": 2.0,
        "tt_start_grain_fill": 660,
        "description": "Uri - 우리"
    },
    "금강": {
        "photop_sens": 2.0,
        "vern_sens": 2.6,
        "tt_start_grain_fill": 680,
        "description": "Keumgang - 금강"
    },
    "조경": {
        "photop_sens": 1.7,
        "vern_sens": 2.0,
        "tt_start_grain_fill": 720,
        "description": "Jogyeong - 조경"
    },
    "사용자 정의": {
        "photop_sens": 1.5,
        "vern_sens": 1.5,
        "tt_start_grain_fill": 545,
        "description": "Custom parameters - 사용자 정의"
    }
}

def create_demo_data():
    """데모 데이터 로드"""
    demo_file = './input/input_weather.csv'
    if os.path.exists(demo_file):
        return pd.read_csv(demo_file)
    return None

def validate_data(df):
    """입력 데이터 유효성 검사"""
    required_columns = ['site', 'year', 'day', 'radn', 'maxt', 'mint', 'rain', 'day_length']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        return False, f"필수 컬럼이 누락되었습니다: {', '.join(missing_columns)}"
    
    # 데이터 타입 검사
    try:
        df['year'] = pd.to_numeric(df['year'])
        df['day'] = pd.to_numeric(df['day'])
        df['maxt'] = pd.to_numeric(df['maxt'])
        df['mint'] = pd.to_numeric(df['mint'])
    except:
        return False, "숫자 데이터에 오류가 있습니다. year, day, maxt, mint 컬럼을 확인하세요."
    
    return True, "데이터가 유효합니다."

def run_wheat_model(df, latitude, sowing_date, stage_params):
    """밀 생육 모델 실행"""
    try:
        # APSIM Wheat Phenology 모델 실행
        apsim_wheat = APSIMWheatPhenology(
            R_p=stage_params['R_p'], 
            R_v=stage_params['R_v'], 
            sowing_date=sowing_date
        )
        results_df = apsim_wheat.accumulate_daily_values(df, latitude)
        
        # 생육 단계 계산
        stage_div = {
            'tt_emergence': stage_params['tt_emergence'],
            'tt_end_of_juvenile': stage_params['tt_end_of_juvenile'],
            'tt_floral_initiation': stage_params['tt_floral_initiation'],
            'tt_flowering': stage_params['tt_flowering'],
            'tt_start_grain_fill': stage_params['tt_start_grain_fill'],
            'tt_end_grain_fill': stage_params['tt_end_grain_fill'],
        }
        
        final_result = wheat_stage_process(results_df, stage_div)
        return final_result, None
        
    except Exception as e:
        return None, f"모델 실행 중 오류가 발생했습니다: {str(e)}"

def create_visualizations(df):
    """결과 시각화"""
    
    # 날짜 변환
    df_plot = df.copy()
    df_plot['Date'] = pd.to_datetime(df_plot['Date'])
    
    # 1. 온도 및 열적 시간 그래프
    fig1 = make_subplots(
        rows=2, cols=1,
        subplot_titles=('일별 온도 변화', '누적 열적 시간'),
        vertical_spacing=0.1
    )
    
    # 온도 그래프
    fig1.add_trace(
        go.Scatter(x=df_plot['Date'], y=df_plot['T_max'], 
                  name='최고온도', line=dict(color='red')),
        row=1, col=1
    )
    fig1.add_trace(
        go.Scatter(x=df_plot['Date'], y=df_plot['T_min'], 
                  name='최저온도', line=dict(color='blue')),
        row=1, col=1
    )
    fig1.add_trace(
        go.Scatter(x=df_plot['Date'], y=df_plot['Crown temperature (T_c)'], 
                  name='관부온도', line=dict(color='green')),
        row=1, col=1
    )
    
    # 누적 열적 시간
    fig1.add_trace(
        go.Scatter(x=df_plot['Date'], y=df_plot['Cumulative_TT'], 
                  name='누적 열적 시간', line=dict(color='orange')),
        row=2, col=1
    )
    
    fig1.update_layout(height=600, title_text="온도 및 열적 시간 변화")
    fig1.update_xaxes(title_text="날짜", row=2, col=1)
    fig1.update_yaxes(title_text="온도 (°C)", row=1, col=1)
    fig1.update_yaxes(title_text="누적 열적 시간", row=2, col=1)
    
    # 2. 생육 단계별 날짜 표시
    stage_columns = ['Emergence_date', 'End_of_juvenile_date', 'floral_initiation_date', 
                    'flowering_date', 'heading_date', 'end_grain_fill_date', 'maturity_date']
    stage_names = ['출아', '유년기 종료', '화아분화', '개화', '출수', '등숙 종료', '성숙']
    
    stage_dates = {}
    for i, col in enumerate(stage_columns):
        if col in df.columns:
            stage_date = df[col].dropna()
            if not stage_date.empty:
                # day of year를 실제 날짜로 변환
                doy = stage_date.iloc[0]
                if not pd.isna(doy):
                    # 첫 번째 년도 기준으로 날짜 계산
                    year = df['Year'].iloc[0]
                    date = datetime(year, 1, 1) + pd.Timedelta(days=int(doy)-1)
                    stage_dates[stage_names[i]] = date
    
    # 3. 광주기 및 저온 처리 인자
    fig2 = make_subplots(
        rows=2, cols=1,
        subplot_titles=('광주기 인자 및 저온 처리 인자', '일장 변화'),
        vertical_spacing=0.1
    )
    
    fig2.add_trace(
        go.Scatter(x=df_plot['Date'], y=df_plot['Photoperiod factor (f_D)'], 
                  name='광주기 인자', line=dict(color='purple')),
        row=1, col=1
    )
    fig2.add_trace(
        go.Scatter(x=df_plot['Date'], y=df_plot['Vernalisation factor (f_V)'], 
                  name='저온 처리 인자', line=dict(color='brown')),
        row=1, col=1
    )
    
    fig2.add_trace(
        go.Scatter(x=df_plot['Date'], y=df_plot['L_p'], 
                  name='일장 시간', line=dict(color='gold')),
        row=2, col=1
    )
    
    fig2.update_layout(height=600, title_text="환경 인자 변화")
    fig2.update_xaxes(title_text="날짜", row=2, col=1)
    fig2.update_yaxes(title_text="인자 값", row=1, col=1)
    fig2.update_yaxes(title_text="일장 (시간)", row=2, col=1)
    
    return fig1, fig2, stage_dates

def main():
    st.title("🌾 한국 밀 생육 단계 예측 시스템")
    st.markdown("""
    **APSIM-Wheat 모델 기반 한국 밀 품종별 생육 시뮬레이션**
    
    🎯 **주요 기능**: 조광, 우리, 금강, 조경 품종별 최적화 파라미터 제공 | 품종 간 생육 특성 비교 | 생육 단계별 예측
    """)
    
    # 품종 선택에 따른 안내 메시지
    if 'selected_cultivar' in locals():
        if selected_cultivar != "사용자 정의":
            st.info(f"✅ 현재 **{KOREAN_WHEAT_CULTIVARS[selected_cultivar]['description']}** 품종으로 설정되었습니다.")
    
    # 사이드바 - 모델 파라미터
    st.sidebar.header("📋 모델 파라미터")
    
    # 품종 선택
    st.sidebar.subheader("🌾 밀 품종 선택")
    selected_cultivar = st.sidebar.selectbox(
        "품종을 선택하세요:",
        list(KOREAN_WHEAT_CULTIVARS.keys()),
        index=0,
        help="한국 밀 품종별로 최적화된 파라미터가 자동 적용됩니다."
    )
    
    # 선택된 품종 정보 표시
    cultivar_info = KOREAN_WHEAT_CULTIVARS[selected_cultivar]
    
    if selected_cultivar != "사용자 정의":
        # 품종별 파라미터 정보를 깔끔하게 표시
        st.sidebar.success(f"🌾 **{cultivar_info['description']}**")
        
        with st.sidebar.expander("📊 품종 파라미터 정보", expanded=False):
            st.write(f"**광주기 반응 (Rp):** {cultivar_info['photop_sens']}")
            st.write(f"**저온 반응 (Rv):** {cultivar_info['vern_sens']}")
            st.write(f"**등숙 시작 열적시간:** {cultivar_info['tt_start_grain_fill']}°C")
    else:
        st.sidebar.info("🔧 **사용자 정의 파라미터**")
    
    st.sidebar.divider()
    
    # 위치 및 파종 정보
    st.sidebar.subheader("🌍 재배 조건")
    latitude = st.sidebar.number_input("위도 (Latitude)", value=35.7281, format="%.4f")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        sowing_month = st.selectbox("파종 월", range(1, 13), index=10)
        sowing_day = st.selectbox("파종 일", range(1, 32), index=4)
    with col2:
        sowing_year = st.number_input("파종 연도", value=1976, min_value=1900, max_value=2100)
    
    sowing_date = datetime(sowing_year, sowing_month, sowing_day)
    
    # 품종별 파라미터 설정
    if selected_cultivar == "사용자 정의":
        R_p = cultivar_info['photop_sens']
        R_v = cultivar_info['vern_sens']
        tt_start_grain_fill = cultivar_info['tt_start_grain_fill']
    else:
        R_p = cultivar_info['photop_sens']
        R_v = cultivar_info['vern_sens']
        tt_start_grain_fill = cultivar_info['tt_start_grain_fill']
    
    # 기본 열적 시간 값들
    tt_emergence = 1.0
    tt_end_of_juvenile = 400.0
    tt_floral_initiation = 555.0
    tt_flowering = 120.0
    tt_end_grain_fill = 35.0
    
    # 사용자 정의 모드에서만 파라미터 조정 가능
    if selected_cultivar == "사용자 정의":
        st.sidebar.divider()
        st.sidebar.subheader("🔧 생육 파라미터 설정")
        
        R_p = st.sidebar.number_input("광주기 반응 (R_p)", value=1.5, min_value=0.0, max_value=5.0, step=0.1, 
                                      help="일장에 대한 민감도 (값이 클수록 일장 변화에 민감)")
        R_v = st.sidebar.number_input("저온 반응 (R_v)", value=1.5, min_value=0.0, max_value=5.0, step=0.1,
                                      help="저온 처리에 대한 민감도 (값이 클수록 저온 요구도가 높음)")
        
        st.sidebar.subheader("📅 생육 단계별 필요 열적 시간")
        
        with st.sidebar.expander("기본 생육 단계", expanded=True):
            tt_emergence = st.number_input("출아 (°C)", value=1.0, min_value=0.0, format="%.1f")
            tt_end_of_juvenile = st.number_input("유년기 종료 (°C)", value=400.0, min_value=0.0, format="%.1f")
            tt_floral_initiation = st.number_input("화아분화 (°C)", value=555.0, min_value=0.0, format="%.1f")
            tt_flowering = st.number_input("개화 (°C)", value=120.0, min_value=0.0, format="%.1f")
        
        with st.sidebar.expander("등숙 단계", expanded=True):
            tt_start_grain_fill = st.number_input("등숙 시작 (°C)", value=545.0, min_value=0.0, format="%.1f")
            tt_end_grain_fill = st.number_input("등숙 종료 (°C)", value=35.0, min_value=0.0, format="%.1f")
    
    stage_params = {
        'R_p': R_p, 'R_v': R_v,
        'tt_emergence': tt_emergence,
        'tt_end_of_juvenile': tt_end_of_juvenile,
        'tt_floral_initiation': tt_floral_initiation,
        'tt_flowering': tt_flowering,
        'tt_start_grain_fill': tt_start_grain_fill,
        'tt_end_grain_fill': tt_end_grain_fill
    }
    
    # 메인 영역
    tab1, tab2, tab3, tab4 = st.tabs(["� 시뮬레이션", "� 결과 분석", "🔍 품종 비교", "� 가이드"])
    
    with tab1:
        st.header("🚀 시뮬레이션 실행")
        
        # 현재 선택된 품종 표시
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            if selected_cultivar != "사용자 정의":
                st.success(f"**선택된 품종:** {KOREAN_WHEAT_CULTIVARS[selected_cultivar]['description']}")
            else:
                st.info("**모드:** 사용자 정의 파라미터")
        
        with col2:
            st.metric("위도", f"{latitude}°N")
        
        with col3:
            st.metric("파종일", sowing_date.strftime("%Y-%m-%d"))
        
        st.divider()
        
        # 데이터 입력 방법 선택
        st.subheader("📁 기상 데이터 선택")
        input_method = st.radio(
            "데이터 입력 방법:",
            ["🎯 데모 데이터 사용", "📂 파일 업로드"],
            horizontal=True
        )
        
        uploaded_data = None
        
        if input_method == "🎯 데모 데이터 사용":
            with st.container():
                st.info("🎯 **데모 데이터 사용** - 한국 부안 지역 1975-1976년 기상 데이터")
                demo_data = create_demo_data()
                if demo_data is not None:
                    uploaded_data = demo_data
                    st.success(f"✅ 데모 데이터 로드 완료 ({len(demo_data):,}일간 데이터)")
                    
                    with st.expander("📊 데이터 미리보기", expanded=False):
                        st.dataframe(demo_data.head(10), use_container_width=True)
                        
                        # 데이터 요약 정보
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("총 데이터 기간", f"{len(demo_data)}일")
                        with col2:
                            st.metric("평균 최고온도", f"{demo_data['maxt'].mean():.1f}°C")
                        with col3:
                            st.metric("평균 최저온도", f"{demo_data['mint'].mean():.1f}°C")
                        with col4:
                            st.metric("총 강수량", f"{demo_data['rain'].sum():.1f}mm")
                else:
                    st.error("❌ 데모 데이터를 찾을 수 없습니다.")
        
        elif input_method == "📂 파일 업로드":
            with st.container():
                st.info("📂 **사용자 파일 업로드** - CSV 형식의 기상 데이터를 업로드하세요")
                
                uploaded_file = st.file_uploader(
                    "기상 데이터 CSV 파일 선택", 
                    type=['csv'],
                    help="필수 컬럼: site, year, day, radn, maxt, mint, rain, day_length"
                )
                
                if uploaded_file is not None:
                    try:
                        uploaded_data = pd.read_csv(uploaded_file)
                        st.success(f"✅ 파일 업로드 성공 ({len(uploaded_data):,}행 데이터)")
                        
                        with st.expander("📊 업로드된 데이터 미리보기", expanded=True):
                            st.dataframe(uploaded_data.head(10), use_container_width=True)
                            
                            # 데이터 요약 정보
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("총 데이터 기간", f"{len(uploaded_data)}일")
                            with col2:
                                st.metric("평균 최고온도", f"{uploaded_data['maxt'].mean():.1f}°C")
                            with col3:
                                st.metric("평균 최저온도", f"{uploaded_data['mint'].mean():.1f}°C")
                            with col4:
                                st.metric("총 강수량", f"{uploaded_data['rain'].sum():.1f}mm")
                                
                    except Exception as e:
                        st.error(f"❌ 파일 읽기 오류: {str(e)}")
        
        # 데이터 검증 및 모델 실행
        if uploaded_data is not None:
            st.divider()
            
            is_valid, message = validate_data(uploaded_data)
            
            if is_valid:
                st.success(f"✅ {message}")
                
                # 모델 실행 섹션
                st.subheader("🎯 시뮬레이션 실행")
                
                # 실행 전 설정 요약
                with st.expander("⚙️ 현재 설정 요약", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**품종 정보:**")
                        st.write(f"- 품종: {selected_cultivar}")
                        st.write(f"- 광주기 반응: {R_p}")
                        st.write(f"- 저온 반응: {R_v}")
                    
                    with col2:
                        st.write("**재배 조건:**")
                        st.write(f"- 위도: {latitude}°N")
                        st.write(f"- 파종일: {sowing_date.strftime('%Y년 %m월 %d일')}")
                        st.write(f"- 데이터 기간: {len(uploaded_data)}일")
                
                # 실행 버튼을 더 눈에 띄게
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("🚀 시뮬레이션 실행", type="primary", use_container_width=True):
                        with st.spinner("🌾 밀 생육 시뮬레이션을 실행하고 있습니다..."):
                            result, error = run_wheat_model(uploaded_data, latitude, sowing_date, stage_params)
                            
                            if error:
                                st.error(f"❌ {error}")
                            else:
                                st.success("🎉 시뮬레이션이 성공적으로 완료되었습니다!")
                                # 세션 상태에 결과 저장
                                st.session_state['model_result'] = result
                                st.session_state['stage_params'] = stage_params
                                st.session_state['selected_cultivar'] = selected_cultivar
                                st.session_state['uploaded_data'] = uploaded_data
                                st.balloons()
                                
                                # 결과 요약 표시
                                with st.container():
                                    st.info("👉 **결과 분석** 탭으로 이동하여 상세 결과를 확인하세요!")
            else:
                st.error(f"❌ {message}")
                
                # 데이터 형식 도움말
                with st.expander("❓ 데이터 형식 도움말"):
                    st.markdown("""
                    **필수 컬럼:** `site`, `year`, `day`, `radn`, `maxt`, `mint`, `rain`, `day_length`
                    
                    - `site`: 지점명 (문자열)
                    - `year`: 연도 (정수)
                    - `day`: 연중 일수 DOY (1-366)
                    - `maxt`, `mint`: 최고/최저기온 (°C)
                    - `radn`: 일사량 (MJ/m²/day)
                    - `rain`: 강수량 (mm)
                    - `day_length`: 일장 (시간)
                    """)
    
    with tab2:
        st.header("📈 시뮬레이션 결과 분석")
        
        if 'model_result' in st.session_state:
            result = st.session_state['model_result']
            selected_cultivar_name = st.session_state.get('selected_cultivar', '알 수 없음')
            
            # 헤더 정보
            st.success(f"🌾 **{selected_cultivar_name}** 품종 시뮬레이션 결과")
            
            # 결과 요약 메트릭
            st.subheader("📊 시뮬레이션 요약")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("시뮬레이션 기간", f"{len(result):,}일", help="총 분석 일수")
            
            with col2:
                max_tt = result['Cumulative_TT'].max()
                st.metric("최대 누적 열적시간", f"{max_tt:.1f}°C", help="전체 기간 동안 누적된 열적 시간")
            
            with col3:
                emergence_date = result['Emergence_date'].dropna()
                if not emergence_date.empty:
                    st.metric("출아일", f"DOY {int(emergence_date.iloc[0])}", help="연중 출아 일수")
                else:
                    st.metric("출아일", "미달성", help="출아 조건 미충족")
            
            with col4:
                avg_temp = (result['T_max'].mean() + result['T_min'].mean()) / 2
                st.metric("평균 기온", f"{avg_temp:.1f}°C", help="전체 기간 평균 기온")
            
            st.divider()
            
            # 생육 단계 정보를 먼저 표시
            fig1, fig2, stage_dates = create_visualizations(result)
            
            if stage_dates:
                st.subheader("🌱 주요 생육 단계 달성 날짜")
                stage_df = pd.DataFrame(list(stage_dates.items()), columns=['생육단계', '달성날짜'])
                stage_df['달성날짜'] = stage_df['달성날짜'].dt.strftime('%m월 %d일')
                
                # 생육 단계를 카드 형태로 표시
                cols = st.columns(len(stage_df))
                for i, (_, row) in enumerate(stage_df.iterrows()):
                    with cols[i]:
                        st.metric(
                            label=f"🌾 {row['생육단계']}", 
                            value=row['달성날짜'],
                            help=f"{row['생육단계']} 단계 달성 예상일"
                        )
            
            st.divider()
            
            # 시각화 그래프
            st.subheader("📈 환경 조건 및 생육 반응 분석")
            
            # 탭으로 그래프 분리
            graph_tab1, graph_tab2 = st.tabs(["🌡️ 온도 및 열적시간", "☀️ 환경 인자"])
            
            with graph_tab1:
                st.plotly_chart(fig1, use_container_width=True, key="temp_chart")
            
            with graph_tab2:
                st.plotly_chart(fig2, use_container_width=True, key="env_chart")
            
            st.divider()
            
            # 결과 다운로드 및 상세 데이터
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📥 결과 다운로드")
                csv = result.to_csv(index=False)
                st.download_button(
                    label="📊 전체 결과 CSV 다운로드",
                    data=csv,
                    file_name=f"wheat_{selected_cultivar_name}_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    type="primary",
                    use_container_width=True
                )
                
                # 간단한 통계 정보
                st.info(f"📁 파일 크기: ~{len(csv)/1024:.1f}KB | {len(result):,}행 데이터")
            
            with col2:
                st.subheader("📋 데이터 구성")
                st.write("**포함된 정보:**")
                st.write("- 일별 기상 데이터")
                st.write("- 열적 시간 계산 결과") 
                st.write("- 생육 단계별 달성일")
                st.write("- 환경 인자 (광주기, 저온처리)")
            
            # 상세 데이터 테이블
            with st.expander("📊 상세 결과 데이터 보기", expanded=False):
                st.dataframe(result, use_container_width=True)
                st.caption(f"총 {len(result):,}행 × {len(result.columns)}열 데이터")
                
        else:
            st.info("👈 먼저 **시뮬레이션** 탭에서 모델을 실행하세요.")
            
            # 시작하기 가이드
            with st.container():
                st.markdown("""
                ### 🚀 시뮬레이션 시작하기
                
                1. **사이드바**에서 원하는 밀 품종을 선택하세요
                2. **시뮬레이션 탭**으로 이동하여 기상 데이터를 업로드하세요
                3. **시뮬레이션 실행** 버튼을 클릭하세요
                4. 이 탭에서 상세한 분석 결과를 확인하세요
                """)
    
    with tab3:
        st.header("🔍 한국 밀 품종별 비교 분석")
        
        st.markdown("""
        **여러 품종의 생육 특성을 동시에 비교하여 품종 간 차이를 분석합니다**
        """)
        
        if 'uploaded_data' not in st.session_state:
            st.warning("👈 먼저 **시뮬레이션** 탭에서 기상 데이터를 준비하고 1회 이상 모델을 실행하세요.")
            
            with st.container():
                st.markdown("""
                ### 🔍 품종 비교 분석이란?
                
                - **조광, 우리, 금강, 조경** 품종의 생육 특성을 동시에 분석
                - 동일한 기상 조건에서 품종별 **생육 단계 달성일** 비교
                - 품종별 **열적 시간 누적 패턴** 비교
                - 각 품종의 **환경 반응 특성** 차이 분석
                """)
        else:
            # 비교할 품종 선택
            st.subheader("비교할 품종 선택")
            compare_cultivars = st.multiselect(
                "비교할 품종들을 선택하세요 (최대 4개):",
                [cult for cult in KOREAN_WHEAT_CULTIVARS.keys() if cult != "사용자 정의"],
                default=["조광", "금강"] if len([cult for cult in KOREAN_WHEAT_CULTIVARS.keys() if cult != "사용자 정의"]) >= 2 else None,
                max_selections=4
            )
            
            if len(compare_cultivars) >= 2:
                if st.button("🚀 품종별 비교 시뮬레이션 실행", type="primary"):
                    comparison_results = {}
                    
                    with st.spinner("품종별 시뮬레이션을 실행하고 있습니다..."):
                        for cultivar in compare_cultivars:
                            cultivar_info = KOREAN_WHEAT_CULTIVARS[cultivar]
                            
                            # 품종별 파라미터로 모델 실행
                            cultivar_stage_params = {
                                'R_p': cultivar_info['photop_sens'],
                                'R_v': cultivar_info['vern_sens'],
                                'tt_emergence': 1.0,
                                'tt_end_of_juvenile': 400.0,
                                'tt_floral_initiation': 555.0,
                                'tt_flowering': 120.0,
                                'tt_start_grain_fill': cultivar_info['tt_start_grain_fill'],
                                'tt_end_grain_fill': 35.0,
                            }
                            
                            result, error = run_wheat_model(
                                st.session_state['uploaded_data'], 
                                latitude, 
                                sowing_date, 
                                cultivar_stage_params
                            )
                            
                            if not error:
                                comparison_results[cultivar] = result
                    
                    if comparison_results:
                        st.success(f"{len(comparison_results)}개 품종의 시뮬레이션이 완료되었습니다!")
                        
                        # 생육 단계별 비교 테이블
                        st.subheader("📅 품종별 생육 단계 비교")
                        
                        stage_comparison = []
                        stage_columns = ['Emergence_date', 'End_of_juvenile_date', 'floral_initiation_date', 
                                       'flowering_date', 'heading_date', 'end_grain_fill_date', 'maturity_date']
                        stage_names = ['출아', '유년기 종료', '화아분화', '개화', '출수', '등숙 종료', '성숙']
                        
                        for cultivar, result in comparison_results.items():
                            row = {'품종': cultivar}
                            for i, col in enumerate(stage_columns):
                                if col in result.columns:
                                    stage_date = result[col].dropna()
                                    if not stage_date.empty:
                                        doy = stage_date.iloc[0]
                                        if not pd.isna(doy):
                                            year = result['Year'].iloc[0]
                                            date = datetime(year, 1, 1) + pd.Timedelta(days=int(doy)-1)
                                            row[stage_names[i]] = date.strftime('%m-%d')
                                        else:
                                            row[stage_names[i]] = "-"
                                    else:
                                        row[stage_names[i]] = "-"
                                else:
                                    row[stage_names[i]] = "-"
                            stage_comparison.append(row)
                        
                        comparison_df = pd.DataFrame(stage_comparison)
                        st.dataframe(comparison_df, use_container_width=True)
                        
                        # 품종별 온도 및 열적 시간 비교 그래프
                        st.subheader("📈 품종별 누적 열적 시간 비교")
                        
                        fig_comparison = go.Figure()
                        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
                        
                        for i, (cultivar, result) in enumerate(comparison_results.items()):
                            result_plot = result.copy()
                            result_plot['Date'] = pd.to_datetime(result_plot['Date'])
                            
                            fig_comparison.add_trace(
                                go.Scatter(
                                    x=result_plot['Date'], 
                                    y=result_plot['Cumulative_TT'],
                                    name=f'{cultivar} ({KOREAN_WHEAT_CULTIVARS[cultivar]["description"]})',
                                    line=dict(color=colors[i % len(colors)], width=2)
                                )
                            )
                        
                        fig_comparison.update_layout(
                            title="품종별 누적 열적 시간 비교",
                            xaxis_title="날짜",
                            yaxis_title="누적 열적 시간",
                            height=500,
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig_comparison, use_container_width=True)
                        
                        # 품종별 파라미터 비교 테이블
                        st.subheader("⚙️ 품종별 파라미터 비교")
                        
                        param_comparison = []
                        for cultivar in compare_cultivars:
                            cultivar_info = KOREAN_WHEAT_CULTIVARS[cultivar]
                            param_comparison.append({
                                '품종': cultivar,
                                '광주기 반응 (Rp)': cultivar_info['photop_sens'],
                                '저온 반응 (Rv)': cultivar_info['vern_sens'],
                                '등숙 시작 열적시간': cultivar_info['tt_start_grain_fill']
                            })
                        
                        param_df = pd.DataFrame(param_comparison)
                        st.dataframe(param_df, use_container_width=True)
                        
                        # 비교 결과 다운로드
                        st.subheader("📥 비교 결과 다운로드")
                        
                        # 모든 품종 결과를 하나의 파일로 합치기
                        combined_results = []
                        for cultivar, result in comparison_results.items():
                            result_copy = result.copy()
                            result_copy['Cultivar'] = cultivar
                            combined_results.append(result_copy)
                        
                        if combined_results:
                            combined_df = pd.concat(combined_results, ignore_index=True)
                            csv_combined = combined_df.to_csv(index=False)
                            st.download_button(
                                label="품종별 비교 결과를 CSV로 다운로드",
                                data=csv_combined,
                                file_name=f"wheat_cultivar_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
            
            elif len(compare_cultivars) == 1:
                st.warning("비교를 위해서는 최소 2개 품종을 선택해주세요.")
            
            # 품종 정보 표시
            if compare_cultivars:
                st.subheader("📋 선택된 품종 정보")
                for cultivar in compare_cultivars:
                    with st.expander(f"{cultivar} - {KOREAN_WHEAT_CULTIVARS[cultivar]['description']}"):
                        info = KOREAN_WHEAT_CULTIVARS[cultivar]
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("광주기 반응 (Rp)", info['photop_sens'])
                        with col2:
                            st.metric("저온 반응 (Rv)", info['vern_sens'])
                        with col3:
                            st.metric("등숙 시작 열적시간", info['tt_start_grain_fill'])
    
    with tab4:
        st.header("� 사용 가이드 및 기술 정보")
        
        st.markdown("""
        ## 🌾 APSIM Wheat Phenology Model 사용법
        
        ### 🌾 한국 밀 품종별 최적화 파라미터
        
        이 모델은 APSIM-Wheat 모델 보정을 통해 도출된 한국 밀 품종별 최적화 파라미터를 제공합니다:
        
        | 품종 | 광주기 반응 (Rp) | 저온 반응 (Rv) | 등숙 시작 열적시간 |
        |------|------------------|----------------|-------------------|
        | 조광 (Jogwang) | 2.6 | 3.8 | 660°C |
        | 우리 (Uri) | 2.3 | 2.0 | 660°C |
        | 금강 (Keumgang) | 2.0 | 2.6 | 680°C |
        | 조경 (Jogyeong) | 1.7 | 2.0 | 720°C |
        
        ### 📋 입력 데이터 형식
        
        업로드하는 CSV 파일은 다음 컬럼들을 포함해야 합니다:
        
        | 컬럼명 | 설명 | 단위 | 예시 |
        |--------|------|------|------|
        | `site` | 지점명 | 문자열 | Buan, Seoul |
        | `year` | 연도 | 정수 | 1975, 2023 |
        | `day` | 연중 일수 (DOY) | 정수 (1-366) | 1, 365 |
        | `radn` | 일사량 | MJ/m²/day | 3.311, 25.5 |
        | `maxt` | 일 최고기온 | °C | 4.3, 32.1 |
        | `mint` | 일 최저기온 | °C | -0.1, 15.2 |
        | `rain` | 강수량 | mm | 0.2, 15.5 |
        | `day_length` | 일장 | 시간 | 10.746, 14.2 |
        
        ### 🔧 모델 파라미터 설명
        
        #### 기본 설정
        - **위도**: 모델링할 지역의 위도 (도 단위)
        - **파종일**: 밀 파종 날짜
        
        #### 생육 파라미터
        - **R_p (광주기 반응)**: 일장에 대한 민감도 (값이 클수록 일장 변화에 민감)
        - **R_v (저온 반응)**: 저온 처리에 대한 민감도 (값이 클수록 저온 요구도가 높음)
        
        #### 품종별 특성
        - **조광**: 저온 요구도가 가장 높고 광주기 반응이 강한 품종
        - **우리**: 저온 요구도가 낮고 광주기 반응이 중간인 품종  
        - **금강**: 균형 잡힌 광주기/저온 반응을 보이는 품종
        - **조경**: 광주기 반응이 가장 약하고 등숙 기간이 긴 품종
        
        #### 생육 단계별 필요 열적 시간
        - **출아**: 파종부터 출아까지 필요한 열적 시간
        - **유년기 종료**: 출아부터 유년기 종료까지
        - **화아분화**: 유년기 종료부터 화아분화까지
        - **개화**: 화아분화부터 개화까지
        - **등숙 시작**: 개화부터 등숙 시작까지
        - **등숙 종료**: 등숙 시작부터 종료까지
        
        ### 📊 결과 해석
        
        모델은 다음과 같은 결과를 제공합니다:
        
        1. **일별 환경 인자**: 온도, 일장, 광주기 인자, 저온 처리 인자
        2. **열적 시간 누적**: 일별 및 누적 열적 시간
        3. **생육 단계별 예상 날짜**: 각 생육 단계 도달 예상일
        4. **시각화 그래프**: 온도 변화, 열적 시간 누적, 환경 인자 변화
        
        ### ⚠️ 주의사항
        
        - 데이터는 연속적이어야 하며, 누락된 날짜가 없어야 합니다.
        - 파종일은 데이터 기간 내에 포함되어야 합니다.
        - 기온 데이터는 섭씨 온도로 입력하세요.
        - 모델은 겨울밀(winter wheat)을 기준으로 합니다.
        
        ### � 사용 방법
        
        1. **품종 선택**: 사이드바에서 시뮬레이션할 한국 밀 품종을 선택
        2. **데이터 입력**: 기상 데이터를 업로드하거나 데모 데이터 사용
        3. **모델 실행**: 단일 품종 분석 또는 여러 품종 비교 실행
        4. **결과 분석**: 생육 단계, 환경 반응, 품종 간 차이 분석
    
        """)
        
        # 품종별 파라미터 비교 테이블
        st.subheader("🌾 품종별 파라미터 요약")
        cultivar_summary = []
        for cultivar, info in KOREAN_WHEAT_CULTIVARS.items():
            if cultivar != "사용자 정의":
                cultivar_summary.append({
                    '품종': cultivar,
                    '영문명': info['description'].split(' - ')[0],
                    '광주기 반응 (Rp)': info['photop_sens'],
                    '저온 반응 (Rv)': info['vern_sens'],
                    '등숙 시작 열적시간': info['tt_start_grain_fill']
                })
        
        summary_df = pd.DataFrame(cultivar_summary)
        st.dataframe(summary_df, use_container_width=True)
        
        # 샘플 데이터 표시
        st.subheader("📄 샘플 데이터 예시")
        sample_data = {
            'site': ['Buan', 'Buan', 'Buan', 'Buan', 'Buan'],
            'year': [1975, 1975, 1975, 1975, 1975],
            'day': [1, 2, 3, 4, 5],
            'radn': [3.311, 4.344, 6.545, 7.387, 8.716],
            'maxt': [4.3, 5.2, 3.8, 4.8, 5.2],
            'mint': [-0.1, 1.4, -3.5, -3.8, -1.8],
            'rain': [0.2, 0.0, 0.0, 0.0, 2.5],
            'day_length': [10.746, 10.755, 10.766, 10.777, 10.789]
        }
        sample_df = pd.DataFrame(sample_data)
        st.dataframe(sample_df)

if __name__ == "__main__":
    main()