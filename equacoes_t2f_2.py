from manim import *
import numpy as np
from manim_slides import Slide

# Para Manim Community (>= 0.18). Rode algo como:
# manim -pqh equacoes_t2f.py EqRMSScene
# e troque o nome da cena conforme desejar.
# 1. Configura o fundo para BRANCO
config.background_color = WHITE 

# 2. Força que todas as classes principais nasçam pretas por padrão
Text.set_default(color=BLACK)
MathTex.set_default(color=BLACK)
Tex.set_default(color=BLACK)
Line.set_default(color=BLACK)
Arrow.set_default(color=BLACK)
# Rectangle, Circle, etc. também se necessário
SurroundingRectangle.set_default(color=BLACK)

# -------------------------
# Helpers Visuais
# -------------------------

class DerivacaoRMS_Tese(Scene):
    def construct(self):
        # --- Configuração de Estilo ---
        # Definindo cores padrão para variáveis para manter consistência
        c_tempo = BLUE      # T e dt
        c_amostras = PINK # N e n
        c_janela = TEAL     # L
        
        # Título fixo, mas menor para não roubar espaço
        titulo = Text("Derivação RMS: Contínuo para Discreto", font_size=36).to_edge(UP)
        self.add(titulo)

        # =================================================================
        # FASE 1: A Definição Contínua
        # =================================================================
        
        eq_cont = MathTex(
            r"I_{\mathrm{RMS}}", r"=", r"\sqrt{", 
            r"\frac{1}{T}", 
            r"\int_0^T i^2(t)\,dt", 
            r"}"
        ).scale(1.2)
        
        eq_cont.set_color_by_tex("T", c_tempo) # T em azul
        eq_cont.set_color_by_tex("dt", c_tempo) # dt em azul

        self.play(Write(eq_cont))
        self.wait(1)

        # Movemos para o topo esquerdo para abrir espaço para as definições
        self.play(eq_cont.animate.scale(0.8).to_corner(UL, buff=1))

        # =================================================================
        # FASE 2: As Regras de Discretização (Lista Lateral)
        # =================================================================
        
        # Vamos criar uma lista de regras à direita
        rule_1 = MathTex(r"1)\,\, T", r"=", r"N", r"\Delta t").scale(0.8)
        rule_1.set_color_by_tex("T", c_tempo)
        rule_1.set_color_by_tex("N", c_amostras)
        rule_1.set_color_by_tex("Delta", c_tempo)

        rule_2 = MathTex(r"2)\,\, \int_0^T", r"\approx", r"\sum_{n=0}^{N-1} \cdot \Delta t").scale(0.8)
        rule_2.set_color_by_tex("Delta", c_tempo)
        rule_2.set_color_by_tex("N", c_amostras)

        # Agrupamos e posicionamos à direita
        rules_group = VGroup(rule_1, rule_2).arrange(DOWN, buff=0.5).to_edge(RIGHT, buff=2)
        
        self.play(Write(rules_group))
        self.wait(1)

        # =================================================================
        # FASE 3: A Substituição e o Corte (O Pulo do Gato)
        # =================================================================
        
        # Texto explicativo rápido
        txt_sub = Text("Substituindo termos...", font_size=24, color=GRAY).move_to(UP)
        self.play(FadeIn(txt_sub))

        # Equação "bruta" com os termos substituídos explicitamente
        # Note que separei o \Delta t no numerador e denominador para poder cortar depois
        eq_raw = MathTex(
            r"I_{\mathrm{RMS}}", r"\approx", r"\sqrt{",
            r"\frac{1}{N \Delta t}",        # 3
            r"\sum_{n=0}^{N-1} i^2[n]",     # 4
            r"\,\Delta t",                  # 5  (note o espaço fino \,)
            r"}"
        ).scale(1.2)
        
        eq_raw.set_color_by_tex("N", c_amostras)
        eq_raw.set_color_by_tex(r"\Delta t", c_tempo)

        # Transição: As regras e a eq contínua se transformam na eq bruta no centro
        self.play(
            FadeOut(eq_cont),
            FadeOut(rules_group),
            FadeOut(txt_sub),
            Write(eq_raw)
        )
        self.wait(1)

        # --- O CORTE MATEMÁTICO ---
        # Identificando visualmente os termos para cortar
        # Índices podem variar, no MathTex acima: 
        # eq_raw[3] é "1 / N dt"
        # eq_raw[5] é "dt"
        
        # Vamos criar linhas de corte (strike lines)
        # Precisamos mirar especificamente no "Delta t" do denominador e do numerador
        
        # (Truque visual: desenhar linhas vermelhas sobre os Delta t)
        # Como MathTex é um vetor de SVGs, podemos pegar coordenadas aproximadas
        # Mas para garantir, faremos linhas diagonais simples nas posições corretas

        # Em vez de tentar acertar o caractere interno, cerca o bloco "\Delta t" inteiro:
        delta_den = eq_raw[3]  # contém "1/(N \Delta t)"
        delta_num = eq_raw[5]  # contém "\Delta t"
        
        corte_den = Line(
            delta_den.get_corner(DL),
            delta_den.get_corner(UR),
            color=RED,
            stroke_width=3
        )

        corte_num = Line(
            delta_num.get_corner(DL),
            delta_num.get_corner(UR),
            color=RED,
            stroke_width=3
        )

        self.play(Create(corte_den), Create(corte_num))
        self.wait(0.5)

        eq_global = MathTex(
            r"I_{\mathrm{RMS}}", r"\approx", r"\sqrt{",
            r"\frac{1}{N}",
            r"\sum_{n=0}^{N-1} i^2[n]",
            r"}"
        ).scale(1.2)
        eq_global.set_color_by_tex("N", c_amostras)

        self.play(
            FadeOut(corte_den), FadeOut(corte_num),
            TransformMatchingTex(eq_raw, eq_global)
        )
        self.wait(1.5)

        # =================================================================
        # FASE 4: Janela Deslizante (O "Grand Finale")
        # =================================================================
        
        # Limpamos a tela completamente para focar no conceito final
        self.play(FadeOut(eq_global), FadeOut(titulo))
        
        final_title = Text("RMS em Janela Deslizante", font_size=40).to_edge(UP)
        self.play(Write(final_title))

        # Fórmula Final Grande
        eq_window = MathTex(
            r"x_{\mathrm{RMS}}[n]",
            r"=",
            r"\sqrt{",
            r"\frac{1}{L}",
            r"\sum_{m=n-L+1}^{n} x^2[m]",
            r"}"
        ).scale(1.5) # Bem grande
        
        eq_window.set_color_by_tex("L", c_janela) # Destaque na variável da janela

        self.play(Write(eq_window))
        self.wait(1)

        # Animação de Retângulo ao redor para finalizar
        frame = SurroundingRectangle(eq_window, color=c_janela, buff=0.3)
        
        # Legenda explicativa
        legenda = VGroup(
            Text("L = Tamanho da Janela (amostras)", font_size=24, color=c_janela),
            Text("n = Amostra atual", font_size=24, color=WHITE)
        ).arrange(DOWN, aligned_edge=LEFT).next_to(frame, DOWN, buff=0.5)

        self.play(Create(frame), Write(legenda))
        self.wait(3)

class Clarke_Tese(Scene):
    def construct(self):
        # --- CONFIGURAÇÕES DE ESTILO ---
        c_a = RED
        c_b = GREEN
        c_c = BLUE
        c_alpha = PINK
        c_beta = ORANGE
        c_corte = RED_E 

        FS_TITLE = 34
        FS_SUBTITLE = 26
        FS_TEXT = 20
        FS_MATH = 36

        EDGE_BUFF_SIDE = 1.5
        EDGE_BUFF_TOP = 0.8

        # Título principal
        main_title = Tex(
            r"Transformação de Clarke: $\mathbf{abc} \rightarrow \boldsymbol{\alpha\beta}$",
            font_size=FS_TITLE
        )
        main_title.to_edge(UP, buff=EDGE_BUFF_TOP)
        self.add(main_title)

        # =========================================================
        # ETAPA 1: GEOMETRIA VETORIAL (abc + eixos αβ)
        # =========================================================
        origin = DOWN * 0.8

        # Fases a, b, c
        vec_a = Arrow(origin, origin + RIGHT*2.5, color=c_a, buff=0)
        lbl_a = MathTex("i_a", color=c_a).next_to(vec_a.get_end(), RIGHT)

        vec_b = Arrow(
            origin,
            origin + 2.5*np.array([np.cos(2*np.pi/3), np.sin(2*np.pi/3), 0]),
            color=c_b,
            buff=0
        )
        lbl_b = MathTex("i_b", color=c_b).next_to(vec_b.get_end(), UP+LEFT, buff=0.1)

        vec_c = Arrow(
            origin,
            origin + 2.5*np.array([np.cos(-2*np.pi/3), np.sin(-2*np.pi/3), 0]),
            color=c_c,
            buff=0
        )
        lbl_c = MathTex("i_c", color=c_c).next_to(vec_c.get_end(), DOWN+LEFT, buff=0.1)

        group_abc = VGroup(vec_a, lbl_a, vec_b, lbl_b, vec_c, lbl_c)

        self.play(
            GrowArrow(vec_a), Write(lbl_a),
            GrowArrow(vec_b), Write(lbl_b),
            GrowArrow(vec_c), Write(lbl_c),
            run_time=1.5
        )
        self.wait(0.5)

        # Eixos α e β
        vec_alpha = Arrow(origin, origin + RIGHT*3.2, color=c_alpha, stroke_width=6, buff=0)
        lbl_alpha = MathTex(r"\alpha", color=c_alpha).next_to(vec_alpha.get_end(), UP)

        vec_beta = Arrow(origin, origin + UP*2.8, color=c_beta, stroke_width=6, buff=0)
        lbl_beta = MathTex(r"\beta", color=c_beta).next_to(vec_beta.get_end(), LEFT)

        group_clarke = VGroup(vec_alpha, lbl_alpha, vec_beta, lbl_beta)

        self.play(FadeIn(group_clarke))
        self.wait(1)

        # =========================================================
        # ETAPA 2: MATRIZ DE CLARKE (forma αβ)
        # =========================================================
        diagrama_full = VGroup(group_abc, group_clarke)

        self.play(
            diagrama_full.animate.scale(0.5).to_edge(LEFT, buff=EDGE_BUFF_SIDE).shift(DOWN*0.5)
        )

        txt_projecao = Text(
            "Projeção nos eixos ortogonais",
            font_size=FS_TEXT,
            color=GRAY
        )
        txt_projecao.next_to(diagrama_full, DOWN, buff=0.3)
        self.play(Write(txt_projecao))

        matrix_tex = MathTex(
            r"\begin{bmatrix} i_\alpha \\ i_\beta \end{bmatrix}",
            r"=",
            r"\frac{2}{3}",
            r"\begin{bmatrix}"
            r"1 & -\frac{1}{2} & -\frac{1}{2} \\"
            r"0 & \frac{\sqrt{3}}{2} & -\frac{\sqrt{3}}{2}"
            r"\end{bmatrix}",
            r"\begin{bmatrix} i_a \\ i_b \\ i_c \end{bmatrix}",
            font_size=FS_MATH
        )
        matrix_tex.to_edge(RIGHT, buff=EDGE_BUFF_SIDE)

        arrow_row1 = Arrow(
            start=matrix_tex.get_top() + UP*0.8,
            end=matrix_tex.get_top() + DOWN*0.1,
            color=c_alpha,
            buff=0.1
        )
        txt_row1 = Text(
            "Linha 1 → projeção em α",
            font_size=FS_TEXT,
            color=c_alpha
        ).next_to(arrow_row1, UP, buff=0.1)

        self.play(Write(matrix_tex))
        self.play(GrowArrow(arrow_row1), FadeIn(txt_row1))
        self.wait(2)

        self.play(
            FadeOut(diagrama_full), FadeOut(txt_projecao),
            FadeOut(matrix_tex), FadeOut(arrow_row1), FadeOut(txt_row1)
        )

        # =========================================================
        # ETAPA 3: DERIVAÇÃO DE i_α
        # =========================================================
        header_alpha = Text(
            "1. Derivando $i_\\alpha$",
            font_size=FS_SUBTITLE,
            color=c_alpha
        )
        header_alpha.to_edge(LEFT, buff=EDGE_BUFF_SIDE).shift(UP*1.5)
        self.play(Write(header_alpha))

        eq_a1 = MathTex(
            r"i_\alpha = \frac{2}{3} \left( 1\cdot i_a - \frac{1}{2} i_b - \frac{1}{2} i_c \right)",
            font_size=FS_MATH+5
        )
        self.play(Write(eq_a1))
        self.wait(1)

        eq_a2 = MathTex(
            r"i_\alpha = \frac{2 i_a - i_b - i_c}{3}",
            font_size=FS_MATH+10,
            color=c_alpha
        )
        self.play(ReplacementTransform(eq_a1, eq_a2))
        self.wait(1)

        eq_a_stored = eq_a2.copy().scale(0.7)
        eq_a_stored.next_to(main_title, DOWN, buff=1).to_edge(LEFT, buff=EDGE_BUFF_SIDE)

        self.play(
            Transform(eq_a2, eq_a_stored),
            FadeOut(header_alpha)
        )

        # =========================================================
        # ETAPA 4: DERIVAÇÃO DE i_β
        # =========================================================
        header_beta = Text(
            "2. Derivando $i_\\beta$",
            font_size=FS_SUBTITLE,
            color=c_beta
        )
        header_beta.to_edge(LEFT, buff=EDGE_BUFF_SIDE)
        self.play(Write(header_beta))

        # Passo 1: produto 2/3 * sqrt(3)/2
        eq_b1 = MathTex(
            r"i_\beta", r"=",
            r"\frac{2}{3} \cdot \frac{\sqrt{3}}{2}",
            r"(i_b - i_c)",
            font_size=FS_MATH+5
        )
        self.play(Write(eq_b1))
        self.wait(1)

        # Passo 2: simplificação para sqrt(3)/3, com risco no bloco todo
        eq_b_mid = MathTex(
            r"i_\beta", r"=",
            r"\frac{\sqrt{3}}{3}",
            r"(i_b - i_c)",
            font_size=FS_MATH+5
        )

        strike = Line(
            eq_b1[2].get_corner(DL),
            eq_b1[2].get_corner(UR),
            color=c_corte,
            stroke_width=4
        )
        self.play(Create(strike))
        self.wait(0.3)
        self.play(FadeOut(strike), TransformMatchingTex(eq_b1, eq_b_mid))
        self.wait(1)

        # Passo 3: forma final 1/sqrt(3)
        eq_b2 = MathTex(
            r"i_\beta = \frac{i_b - i_c}{\sqrt{3}}",
            font_size=FS_MATH+10,
            color=c_beta
        )
        self.play(TransformMatchingTex(eq_b_mid, eq_b2))
        self.wait(1)

        # =========================================================
        # ETAPA 5: RESUMO FINAL
        # =========================================================
        self.play(FadeOut(header_beta))

        final_alpha = eq_a2
        final_beta = eq_b2

        grupo_formulas = VGroup(final_alpha, final_beta).arrange(DOWN, buff=0.8)

        self.play(
            final_alpha.animate.scale(1.4),
            grupo_formulas.animate.move_to(ORIGIN)
        )

        box = SurroundingRectangle(grupo_formulas, color=WHITE, buff=0.5, corner_radius=0.2)
        lbl_final = Text(
            "Formas finais da Transformação de Clarke",
            font_size=FS_TEXT
        ).next_to(box, DOWN)

        self.play(Create(box), Write(lbl_final))
        self.wait(3)
    
class EqSeq_Tese(Scene):
    def construct(self):
        # --- CONFIGURAÇÕES VISUAIS ---
        # Cores
        c_a = RED
        c_b = GREEN
        c_c = BLUE
        c_zero = PINK
        c_pos = TEAL
        c_neg = PURPLE
        
        # Tamanhos
        FS_TITLE = 34
        FS_SUBTITLE = 24
        FS_MATH = 38
        FS_TEXT = 20
        
        # Título Principal (Fixo)
        title = Title("Componentes Simétricas (Fortescue)", font_size=FS_TITLE)
        self.add(title)

        # =================================================================
        # ETAPA 1: O OPERADOR 'a'
        # =================================================================
        
        # Cabeçalho Fixo
        header_1 = Text("1. O Operador de Rotação 'a'", font_size=FS_SUBTITLE, color=GRAY)
        header_1.to_edge(UP, buff=1.2).to_edge(LEFT, buff=1)
        self.play(Write(header_1))

        # Configuração dos Vetores
        origin_op = LEFT * 2.5 + DOWN * 0.5
        radius = 1.5
        
        # Círculo auxiliar
        circle = Circle(radius=radius, color=WHITE, stroke_opacity=0.2).move_to(origin_op)
        
        # Vetor 1
        v1 = Arrow(origin_op, origin_op + RIGHT*radius, buff=0, color=WHITE, stroke_width=4)
        l1 = MathTex("1", font_size=FS_TEXT).next_to(v1.get_end(), RIGHT, buff=0.1)
        
        # Vetor a
        pos_a = origin_op + radius*np.array([np.cos(2*np.pi/3), np.sin(2*np.pi/3), 0])
        v_a = Arrow(origin_op, pos_a, buff=0, color=c_pos, stroke_width=4)
        l_a = MathTex("a", font_size=FS_TEXT, color=c_pos).next_to(v_a.get_end(), UP+LEFT, buff=0.1)
        
        # Vetor a^2
        pos_a2 = origin_op + radius*np.array([np.cos(4*np.pi/3), np.sin(4*np.pi/3), 0])
        v_a2 = Arrow(origin_op, pos_a2, buff=0, color=c_neg, stroke_width=4)
        l_a2 = MathTex("a^2", font_size=FS_TEXT, color=c_neg).next_to(v_a2.get_end(), DOWN+LEFT, buff=0.1)

        # Animação
        self.play(Create(circle), GrowArrow(v1), Write(l1))
        self.wait(0.5)
        self.play(GrowArrow(v_a), Write(l_a))
        self.play(GrowArrow(v_a2), Write(l_a2))

        # Definições Matemáticas (Lado Direito, bem separado)
        defs = VGroup(
            MathTex(r"a = 1\angle 120^\circ", font_size=FS_MATH),
            MathTex(r"a^2 = 1\angle 240^\circ", font_size=FS_MATH),
            MathTex(r"1 + a + a^2 = 0", font_size=FS_MATH, color=PINK)
        ).arrange(DOWN, buff=0.5).to_edge(RIGHT, buff=2)

        self.play(Write(defs))
        self.wait(2)

        # LIMPEZA TOTAL (Para evitar sobreposição na próxima cena)
        self.play(
            FadeOut(circle), FadeOut(v1), FadeOut(l1), 
            FadeOut(v_a), FadeOut(l_a), FadeOut(v_a2), FadeOut(l_a2),
            FadeOut(defs), FadeOut(header_1)
        )

        # =================================================================
        # ETAPA 2: AS TRÊS SEQUÊNCIAS (Diagramas lado a lado)
        # =================================================================

        header_2 = Text("2. Decomposição Visual", font_size=FS_SUBTITLE, color=GRAY)
        header_2.to_edge(UP, buff=1.2).to_edge(LEFT, buff=1)
        self.play(Write(header_2))

        # Parâmetros para os mini-gráficos
        r_small = 1.0
        
        # --- GRUPO ZERO (Esquerda) ---
        # Três vetores verticais paralelos
        origin_0 = ORIGIN 
        # Criamos os vetores centrados na origem local (0,0,0) depois movemos o grupo
        v0_a = Arrow(DOWN*0.5, UP*0.5, color=c_zero, buff=0)
        v0_b = Arrow(DOWN*0.5 + RIGHT*0.3, UP*0.5 + RIGHT*0.3, color=c_zero, buff=0)
        v0_c = Arrow(DOWN*0.5 + LEFT*0.3, UP*0.5 + LEFT*0.3, color=c_zero, buff=0)
        lbl_0 = Text("Sequência Zero", font_size=18, color=c_zero).next_to(v0_a, DOWN, buff=0.5)
        lbl_v0_math = MathTex("V_0", font_size=24, color=c_zero).next_to(lbl_0, DOWN, buff=0.1)
        
        grp_0 = VGroup(v0_a, v0_b, v0_c, lbl_0, lbl_v0_math)

        # --- GRUPO POSITIVO (Centro) ---
        # A(0), B(-120), C(120)
        vp_a = Arrow(ORIGIN, RIGHT*r_small, color=c_a, buff=0)
        vp_b = Arrow(ORIGIN, r_small*np.array([np.cos(4*np.pi/3), np.sin(4*np.pi/3), 0]), color=c_b, buff=0)
        vp_c = Arrow(ORIGIN, r_small*np.array([np.cos(2*np.pi/3), np.sin(2*np.pi/3), 0]), color=c_c, buff=0)
        lbl_p = Text("Sequência Positiva", font_size=18, color=c_pos).next_to(vp_b, DOWN, buff=0.5)
        lbl_vp_math = MathTex("V_1", font_size=24, color=c_pos).next_to(lbl_p, DOWN, buff=0.1)

        grp_1 = VGroup(vp_a, vp_b, vp_c, lbl_p, lbl_vp_math)

        # --- GRUPO NEGATIVO (Direita) ---
        # A(0), C(-120), B(120) -> Troca B e C
        vn_a = Arrow(ORIGIN, RIGHT*r_small, color=c_a, buff=0)
        vn_b = Arrow(ORIGIN, r_small*np.array([np.cos(2*np.pi/3), np.sin(2*np.pi/3), 0]), color=c_b, buff=0) # B foi para cima
        vn_c = Arrow(ORIGIN, r_small*np.array([np.cos(4*np.pi/3), np.sin(4*np.pi/3), 0]), color=c_c, buff=0) # C foi para baixo
        lbl_n = Text("Sequência Negativa", font_size=18, color=c_neg).next_to(vn_c, DOWN, buff=0.5)
        lbl_vn_math = MathTex("V_2", font_size=24, color=c_neg).next_to(lbl_n, DOWN, buff=0.1)

        grp_2 = VGroup(vn_a, vn_b, vn_c, lbl_n, lbl_vn_math)

        # ARRANJO GERAL (O segredo para não sobrepor)
        # Coloca os 3 grupos em uma linha horizontal, espaçados
        all_plots = VGroup(grp_0, grp_1, grp_2).arrange(RIGHT, buff=1.5)
        
        self.play(FadeIn(all_plots))
        self.wait(3)

        # LIMPEZA TOTAL
        self.play(FadeOut(all_plots), FadeOut(header_2))

        # =================================================================
        # ETAPA 3: MATRIZ (Centralizada e Limpa)
        # =================================================================

        header_3 = Text("3. Matriz de Transformação", font_size=FS_SUBTITLE, color=GRAY)
        header_3.to_edge(UP, buff=1.2).to_edge(LEFT, buff=1)
        self.play(Write(header_3))

        # Equação Matricial
        # Usando cores por Tex para garantir
        matrix_eq = MathTex(
            r"\begin{bmatrix} V_0 \\ V_1 \\ V_2 \end{bmatrix}",
            r"=",
            r"\frac{1}{3}",
            r"\begin{bmatrix} 1 \& 1 \& 1 \\ 1 \& a \& a^2 \\ 1 \& a^2 \& a \end{bmatrix}",
            r"\begin{bmatrix} V_a \\ V_b \\ V_c \end{bmatrix}",
            font_size=FS_MATH+5
        )

        # Aplicando cores
        matrix_eq.set_color_by_tex("V_0", c_zero)
        matrix_eq.set_color_by_tex("V_1", c_pos)
        matrix_eq.set_color_by_tex("V_2", c_neg)
        matrix_eq.set_color_by_tex("V_a", c_a)
        matrix_eq.set_color_by_tex("V_b", c_b)
        matrix_eq.set_color_by_tex("V_c", c_c)

        self.play(Write(matrix_eq))
        self.wait(2)

        # Destaque didático
        rect = SurroundingRectangle(matrix_eq[3], color=WHITE, buff=0.15)
        lbl_matrix = Text("Matriz de Fortescue Inversa (A^-1)", font_size=16, color=GRAY).next_to(rect, UP)
        
        self.play(Create(rect), FadeIn(lbl_matrix))
        self.wait(2)

        # LIMPEZA TOTAL
        self.play(FadeOut(matrix_eq), FadeOut(rect), FadeOut(lbl_matrix), FadeOut(header_3))

        # =================================================================
        # ETAPA 4: EQUAÇÕES FINAIS (Centralizadas)
        # =================================================================

        header_4 = Text("4. Equações Escalares", font_size=FS_SUBTITLE, color=GRAY)
        header_4.to_edge(UP, buff=1.2).to_edge(LEFT, buff=1)
        self.play(Write(header_4))

        # Equações
        # Usando substring colorido manualmente para precisão
        eq0 = MathTex(r"V_0 = \frac{1}{3}(V_a + V_b + V_c)", font_size=FS_MATH)
        eq0.set_color_by_tex("V_0", c_zero)

        eq1 = MathTex(r"V_1 = \frac{1}{3}(V_a + a V_b + a^2 V_c)", font_size=FS_MATH)
        eq1.set_color_by_tex("V_1", c_pos)
        
        eq2 = MathTex(r"V_2 = \frac{1}{3}(V_a + a^2 V_b + a V_c)", font_size=FS_MATH)
        eq2.set_color_by_tex("V_2", c_neg)

        # Agrupar e alinhar verticalmente com espaço generoso
        eqs = VGroup(eq0, eq1, eq2).arrange(DOWN, buff=0.7)
        
        self.play(Write(eqs))
        self.wait(3)

        # Finalização
        box = SurroundingRectangle(eqs, color=WHITE, buff=0.5)
        self.play(Create(box))
        self.wait(2)


class EqSeqInst_Tese(Scene):
    def construct(self):
        # --- CONFIGURAÇÕES VISUAIS (PADRÃO TESE) ---
        # Cores
        c_sig = PINK     # Sinal Original (Analítico)
        c_rot = RED        # Referencial Girante (Complex Envelope)
        c_res = TEAL       # Resultado (Fasor Estático)
        
        # Tamanhos
        FS_TITLE = 34
        FS_SUBTITLE = 24
        FS_MATH = 36
        FS_TEXT = 20
        EDGE_BUFF = 1.0
        
        # Título Principal
        title = Title("Fasor Instantâneo: A Técnica do Envelope Complexo", font_size=FS_TITLE)
        self.add(title)

        # =================================================================
        # ETAPA 1: O PROBLEMA (O Fasor Girante)
        # =================================================================
        
        # Cabeçalho
        txt_1 = Text("1. O Sinal Analítico (Vetor Girante)", font_size=FS_SUBTITLE, color=GRAY)
        txt_1.to_edge(UP, buff=1.2).to_edge(LEFT, buff=EDGE_BUFF)
        self.play(Write(txt_1))

        # Definição Matemática
        # x_tilde(t) = A * e^(j(wt + phi))
        eq_analytic = MathTex(
            r"\tilde{x}(t)", r"=", r"A", r"e^{j(\omega_0 t + \varphi)}",
            font_size=FS_MATH
        )
        eq_analytic.set_color_by_tex("tilde", c_sig)
        eq_analytic.to_edge(LEFT, buff=EDGE_BUFF).shift(UP*0.5)
        
        self.play(Write(eq_analytic))

        # --- Visualização Geométrica 1 ---
        origin_1 = RIGHT * 3 + DOWN * 0.5
        radius = 1.5
        
        # Eixos e Círculo
        axes_1 = Axes(
            x_range=[-2, 2], y_range=[-2, 2], 
            x_length=4, y_length=4, 
            axis_config={"include_tip": False, "color": GRAY}
        ).move_to(origin_1)
        circle_1 = Circle(radius=radius, color=GRAY, stroke_opacity=0.3).move_to(origin_1)
        
        # O Vetor Girante (Sinal)
        # Vamos simular ele em um instante t qualquer
        angle_sig = np.pi/3 # 60 graus
        vec_sig = Arrow(origin_1, origin_1 + radius*np.array([np.cos(angle_sig), np.sin(angle_sig), 0]), buff=0, color=c_sig, stroke_width=4)
        lbl_sig = MathTex(r"\omega_0", color=c_sig, font_size=24).next_to(vec_sig.get_end(), UP+RIGHT, buff=0.1)
        
        # Arco indicando rotação
        arc_sig = Arc(radius=0.5, start_angle=0, angle=angle_sig, arc_center=origin_1, color=c_sig)
        
        group_fig1 = VGroup(axes_1, circle_1, vec_sig, lbl_sig, arc_sig)
        
        self.play(Create(axes_1), Create(circle_1))
        self.play(GrowArrow(vec_sig), Write(lbl_sig), Create(arc_sig))
        self.wait(1)
        
        # Pequena animação de rotação para mostrar que é instável
        self.play(
            Rotate(vec_sig, angle=np.pi/2, about_point=origin_1),
            Rotate(arc_sig, angle=np.pi/2, about_point=origin_1),
            run_time=2
        )
        self.wait(1)

        # =================================================================
        # ETAPA 2: A SOLUÇÃO (O Referencial Girante)
        # =================================================================

        # Limpa Etapa 1
        self.play(FadeOut(group_fig1), FadeOut(eq_analytic), FadeOut(txt_1))

        txt_2 = Text("2. Removendo a Rotação (Demodulação)", font_size=FS_SUBTITLE, color=GRAY)
        txt_2.to_edge(UP, buff=1.2).to_edge(LEFT, buff=EDGE_BUFF)
        self.play(Write(txt_2))

        # Equação do Referencial
        eq_rot = MathTex(
            r"\text{rot}(t) = e^{-j\omega_0 t}",
            font_size=FS_MATH
        )
        eq_rot.set_color(c_rot)
        eq_rot.to_edge(LEFT, buff=EDGE_BUFF).shift(UP*1)

        self.play(Write(eq_rot))

        # Visualização: Dois vetores
        # Um girando pra esquerda (+w), outro pra direita (-w)
        
        origin_2 = ORIGIN + DOWN * 0.5
        
        # Vetor Sinal (Original)
        vec_s = Arrow(origin_2, origin_2 + radius*np.array([np.cos(np.pi/3), np.sin(np.pi/3), 0]), buff=0, color=c_sig)
        lbl_s = MathTex(r"+ \omega_0", color=c_sig, font_size=24).next_to(vec_s.get_end(), UP)
        
        # Vetor Rotação (Inverso)
        vec_r = Arrow(origin_2, origin_2 + radius*np.array([np.cos(-np.pi/4), np.sin(-np.pi/4), 0]), buff=0, color=c_rot)
        lbl_r = MathTex(r"- \omega_0", color=c_rot, font_size=24).next_to(vec_r.get_end(), DOWN)

        group_vecs = VGroup(vec_s, lbl_s, vec_r, lbl_r)
        
        self.play(GrowArrow(vec_s), Write(lbl_s))
        self.play(GrowArrow(vec_r), Write(lbl_r))
        self.wait(1)
        
        # Animação Conceitual: Girando em sentidos opostos
        self.play(
            Rotate(vec_s, angle=np.pi/2, about_point=origin_2),
            Rotate(vec_r, angle=-np.pi/2, about_point=origin_2), # Gira ao contrário
            run_time=2
        )
        
        # Texto de conclusão visual
        txt_cancel = Text("As velocidades angulares se cancelam!", font_size=20, color=PINK).next_to(origin_2, DOWN, buff=2)
        self.play(Write(txt_cancel))
        self.wait(2)

        # =================================================================
        # ETAPA 3: A MATEMÁTICA E O FILTRO
        # =================================================================

        self.play(FadeOut(group_vecs), FadeOut(eq_rot), FadeOut(txt_2), FadeOut(txt_cancel))

        txt_3 = Text("3. Equação do Fasor Instantâneo", font_size=FS_SUBTITLE, color=GRAY)
        txt_3.to_edge(UP, buff=1.2).to_edge(LEFT, buff=EDGE_BUFF)
        self.play(Write(txt_3))

        # Dedução rápida
        # x(t)*rot(t) = A*e^(jwt+phi) * e^(-jwt)
        step1 = MathTex(
            r"z(t)", r"=", r"\left( A e^{j(\omega_0 t + \varphi)} \right)", r"\cdot", r"\left( e^{-j\omega_0 t} \right)",
            font_size=FS_MATH
        )
        step1[2].set_color(c_sig)
        step1[4].set_color(c_rot)
        
        self.play(Write(step1))
        self.wait(1.5)

        # Simplificação (Corte das exponenciais)
        # Mostra explicitamente que wt corta com -wt
        step2 = MathTex(
            r"z(t)", r"=", r"A", r"e^{j(\varphi + \omega_0 t - \omega_0 t)}",
            font_size=FS_MATH
        )
        step2.set_color_by_tex("varphi", c_res)
        
        self.play(ReplacementTransform(step1, step2))
        self.wait(1)
        
        # Resultado Estático
        step3 = MathTex(
            r"z(t) = A e^{j\varphi} = \text{Constante (DC)}",
            font_size=FS_MATH
        )
        step3.set_color(c_res)
        
        self.play(ReplacementTransform(step2, step3))
        self.wait(2)

        # Move para cima para dar espaço à fórmula final
        self.play(step3.animate.scale(0.8).next_to(txt_3, DOWN, buff=1).to_edge(LEFT, buff=EDGE_BUFF))

        # --- A Fórmula Final (Com Filtro LPF) ---
        # Explicar que na prática precisamos do filtro
        
        final_eq = MathTex(
            r"X_{\mathrm{inst}}(t)", r"=", r"\sqrt{2}", r"\cdot", 
            r"\mathrm{LPF}", r"\left\{", r"x(t)", r"e^{-j\omega_0 t}", r"\right\}",
            font_size=FS_MATH+5
        )
        final_eq.set_color_by_tex("X", c_res)
        final_eq.set_color_by_tex("LPF", BLUE)
        final_eq.set_color_by_tex("omega", c_rot)
        
        final_eq.move_to(ORIGIN)
        
        self.play(Write(final_eq))
        self.wait(1)

        # Destaque das partes
        # 1. Referencial Girante
        frame_rot = SurroundingRectangle(final_eq[7], color=c_rot, buff=0.1)
        lbl_rot = Text("Remove Rotação (Demodulação)", font_size=16, color=c_rot).next_to(frame_rot, DOWN)
        
        self.play(Create(frame_rot), Write(lbl_rot))
        self.wait(1.5)
        
        # 2. Filtro LPF
        frame_lpf = SurroundingRectangle(final_eq[4], color=BLUE, buff=0.1)
        lbl_lpf = Text("Média na Janela / Remove 2w", font_size=16, color=BLUE).next_to(frame_lpf, UP)
        
        self.play(ReplacementTransform(frame_rot, frame_lpf), ReplacementTransform(lbl_rot, lbl_lpf))
        self.wait(2)

        # Limpeza final
        self.play(FadeOut(frame_lpf), FadeOut(lbl_lpf))
        
        # Box final
        box = SurroundingRectangle(final_eq, color=WHITE, buff=0.4, corner_radius=0.2)
        lbl_final = Text("Base para PMU e Proteção Digital", font_size=FS_TEXT).next_to(box, DOWN)
        
        self.play(Create(box), Write(lbl_final))
        self.wait(3)


class EqTCC_Tese(Scene):
    """
    Cena didática para Tese:
    - Define M = I/Ip
    - IEC 60255 (IDMT) vs IEEE C37.112
    """
    def construct(self):
        # =========================================================
        # CONFIGURAÇÕES GERAIS
        # =========================================================
        # self.camera.background_color = "#0F172A" # Slate 900 (Azul Profundo Profissional)
        
        # Cores Unificadas para Clareza
        COLOR_IEC = TEAL
        COLOR_IEEE = PURPLE

        # Eixos e Grid (Serão usados nos momentos de gráfico)
        axes = Axes(
            x_range=[0, 20, 2],
            y_range=[0, 10, 2],
            x_length=8.5,
            y_length=6,
            axis_config={"include_numbers": True, "font_size": 10, "color": LIGHT_GREY},
            tips=False
        )
        
        # --- CORREÇÃO 2: Rótulos reposicionados (t(s) na esquerda, M centralizado embaixo) ---
        # M centralizado abaixo do eixo X
        x_label = axes.get_x_axis_label(MathTex("M"), edge=DOWN, direction=DOWN, buff=0.3)
        # t(s) na esquerda do eixo Y
        y_label = axes.get_y_axis_label(MathTex("t(s)"), edge=LEFT, direction=LEFT, buff=0.8)
        
        labels = VGroup(x_label, y_label)
        
        # Grid mais sutil
        grid = NumberPlane(
            x_range=[0, 20, 2], y_range=[0, 10, 2],
            x_length=10, y_length=6,
            background_line_style={"stroke_opacity": 0.1, "stroke_color": GREY}
        ).move_to(axes)
        
        graph_group = VGroup(grid, axes, labels).move_to(ORIGIN)

        # =========================================================
        # PARTE 1: INTRODUÇÃO TEÓRICA (O CONCEITO DE M)
        # =========================================================
        title_intro = Text("Fundamento: O Múltiplo de Corrente", font_size=36).to_edge(UP)
        
        # Definição matemática grande e clara
        m_def = MathTex(r"M = \frac{I_{\text{medida}}}{I_{\text{pickup}}}", font_size=70)
        
        # Explicação
        explanation = VGroup(
            Text("• I_medida: Corrente de falta real passando no relé", font_size=24, color=GRAY_B),
            Text("• I_pickup: Ajuste de corrente para iniciar atuação", font_size=24, color=GRAY_B),
            Text("• Se M > 1: O relé começa a contar tempo.", font_size=24, color=PINK)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).next_to(m_def, DOWN, buff=0.5)

        self.play(Write(title_intro))
        self.play(Write(m_def))
        self.wait(0.5)
        self.play(FadeIn(explanation, shift=UP*0.2))
        self.wait(3)

        # LIMPEZA TOTAL PARA COMEÇAR AS CURVAS
        self.play(
            FadeOut(title_intro),
            FadeOut(m_def),
            FadeOut(explanation)
        )

        # =========================================================
        # PARTE 2: IEC 60255 (MODO LOUSA -> MODO GRÁFICO)
        # =========================================================
        
        # --- 2.1 MODO LOUSA (Matemática) ---
        title_iec = Text("1. Família IEC 60255 (Standard)", font_size=32, color=COLOR_IEC).to_edge(UP)
        
        eq_iec = MathTex(r"t(M) = TMS \cdot \frac{k}{M^{\alpha} - 1}", font_size=60)
        # Centralizado, mas abrindo espaço em cima para o exemplo
        eq_iec.move_to(ORIGIN) 

        # Tabela IEC
        IEC_PARAMS = [
            ("Standard Inverse",  0.14, 0.02),
            ("Very Inverse",     13.50, 1.00),
            ("Extremely Inverse", 80.00, 2.00),
            ("Long Time",       120.00, 1.00),
            ("Short Time",        0.05, 0.04),
        ]
        
        # Criação manual da tabela para ficar bonita
        t_vals = VGroup()
        for i, (name, k, a) in enumerate(IEC_PARAMS):
            row = Text(f"{name}: k={k}, α={a}", font_size=20, color=GRAY_C)
            t_vals.add(row)
        t_vals.arrange(DOWN, aligned_edge=LEFT).next_to(eq_iec, DOWN, buff=1)

        self.play(Write(title_iec), Write(eq_iec))
        self.play(FadeIn(t_vals))
        self.wait(1.5)

        # --- CORREÇÃO 1: Exemplo movido para CIMA da equação ---
        demo_txt = Text("Exemplo: Very Inverse (k=13.5, α=1, TMS=1)", font_size=24, color=PINK)
        demo_txt.next_to(eq_iec, UP, buff=0.7) # Posicionado acima
        
        step1 = MathTex(r"t = 1 \cdot \frac{13.5}{M^{1} - 1}", font_size=60).move_to(eq_iec)
        
        self.play(Write(demo_txt))
        self.play(Transform(eq_iec, step1))
        self.wait(2)

        # --- TRANSIÇÃO: SAI LOUSA, ENTRA GRÁFICO ---
        self.play(
            FadeOut(eq_iec), FadeOut(t_vals), FadeOut(demo_txt), FadeOut(title_iec)
        )
        
        # --- 2.2 MODO GRÁFICO ---
        self.play(FadeIn(graph_group))
        
        # Título discreto no canto
        label_iec_graph = Text("Curvas IEC", font_size=24, color=COLOR_IEC).to_corner(UL)
        self.play(FadeIn(label_iec_graph))

        iec_curves = VGroup()
        for i, (name, k, a) in enumerate(IEC_PARAMS):
            # Clip function inline
            c = axes.plot(
                lambda m, k=k, a=a: 10 if (m**a - 1) == 0 else min(10, max(0, 1.0*(k/(m**a - 1)))),
                x_range=[1.05, 20], color=COLOR_IEC, stroke_width=2
            )
            # Label na curva
            l = Text(name, font_size=16, color=COLOR_IEC).next_to(c.get_end(), RIGHT)
            iec_curves.add(VGroup(c, l))

        self.play(LaggedStart(*[Create(g[0]) for g in iec_curves], lag_ratio=0.1), run_time=3)
        self.play(FadeIn(VGroup(*[g[1] for g in iec_curves])))
        self.wait(2)

        # LIMPEZA PARA O PRÓXIMO ATO
        self.play(FadeOut(iec_curves), FadeOut(graph_group), FadeOut(label_iec_graph))


        # =========================================================
        # PARTE 3: IEEE C37.112 (MODO LOUSA -> MODO GRÁFICO)
        # =========================================================

        # --- 3.1 MODO LOUSA ---
        title_ieee = Text("2. Família IEEE C37.112", font_size=32, color=COLOR_IEEE).to_edge(UP)
        
        eq_ieee = MathTex(r"t(M) = TD \left( \frac{A}{M^p - 1} + B \right)", font_size=55)
        eq_ieee.move_to(ORIGIN) # Centralizado

        IEEE_PARAMS = [
            ("Moderately Inv", 0.0515, 0.1140, 0.02),
            ("Very Inverse",   19.610, 0.4910, 2.00),
            ("Extremely Inv",  28.200, 0.1217, 2.00),
            ("Inverse (CO8)",   5.950, 0.1800, 2.00),
            ("Short Time",      0.0239, 0.0169, 0.02),
        ]

        # Lista de parâmetros simplificada
        t_vals_ieee = VGroup()
        for i, (name, A, B, p) in enumerate(IEEE_PARAMS):
            row = Text(f"{name}", font_size=24, color=GRAY_C)
            t_vals_ieee.add(row)
        
        desc_ieee = Text("Parâmetros A, B e p definem a curvatura e offset", font_size=24, color=GRAY).next_to(eq_ieee, DOWN, buff=0.7)
        t_vals_ieee.arrange(RIGHT, buff=0.5).next_to(desc_ieee, DOWN)

        self.play(Write(title_ieee), Write(eq_ieee))
        self.play(FadeIn(desc_ieee), FadeIn(t_vals_ieee))
        self.wait(1.5)

        # Demonstração passo a passo IEEE
        # Também movido para cima para manter o padrão
        demo_txt_ieee = Text("Exemplo: Moderately Inv (A=0.0515, B=0.114, p=0.02, TD=1)", font_size=24, color=PINK)
        demo_txt_ieee.next_to(eq_ieee, UP, buff=0.7)
        self.play(Write(demo_txt_ieee))
        
        # Passo 1: Substituição dos valores
        step1_ieee = MathTex(r"t = 1 \cdot \left( \frac{0.0515}{M^{0.02} - 1} + 0.114 \right)", font_size=55).move_to(eq_ieee)
        self.play(Transform(eq_ieee, step1_ieee))
        self.wait(1.5)

        # Passo 2: Destaque para o termo B (Offset)
        offset_desc = Text("O termo '+ B' garante um tempo mínimo de atuação", font_size=24, color=PINK).next_to(eq_ieee, DOWN, buff=0.3)
        self.play(Write(offset_desc))
        self.play(Indicate(eq_ieee[0][14:], color=PINK)) # Indica a parte "+ 0.114 )"
        self.wait(2)

        # --- TRANSIÇÃO ---
        self.play(
            FadeOut(eq_ieee), FadeOut(desc_ieee), FadeOut(t_vals_ieee), FadeOut(demo_txt_ieee), FadeOut(title_ieee), FadeOut(offset_desc)
        )

        # --- 3.2 MODO GRÁFICO ---
        self.play(FadeIn(graph_group))
        
        label_ieee_graph = Text("Curvas IEEE", font_size=24, color=COLOR_IEEE).to_corner(UL)
        self.play(FadeIn(label_ieee_graph))

        ieee_curves = VGroup()
        for i, (name, A, B, p) in enumerate(IEEE_PARAMS):
            c = axes.plot(
                lambda m, A=A, B=B, p=p: 10 if (m**p - 1) == 0 else min(10, max(0, 1.0*((A/(m**p - 1)) + B))),
                x_range=[1.05, 20], color=COLOR_IEEE, stroke_width=2
            )
            l = Text(name, font_size=16, color=COLOR_IEEE).next_to(c.get_end(), RIGHT)
            ieee_curves.add(VGroup(c, l))

        self.play(LaggedStart(*[Create(g[0]) for g in ieee_curves], lag_ratio=0.1), run_time=3)
        self.play(FadeIn(VGroup(*[g[1] for g in ieee_curves])))
        self.wait(2)
        
        # NÃO APAGA O GRÁFICO DESSA VEZ, SÓ AS CURVAS PARA A COMPARAÇÃO
        self.play(FadeOut(ieee_curves), FadeOut(label_ieee_graph))


        # =========================================================
        # PARTE 4: COMPARAÇÃO FINAL (TODAS JUNTAS)
        # =========================================================
        
        final_title = Text("Comparativo Final", font_size=32).to_corner(UL)
        self.play(Write(final_title))

        # Recriar as linhas (sem texto para não poluir) para plotar tudo junto
        all_curves = VGroup()
        
        # IEC (Teal)
        for i, (name, k, a) in enumerate(IEC_PARAMS):
            c = axes.plot(lambda m, k=k, a=a: min(10, max(0, 1.0*(k/(m**a - 1)))), x_range=[1.05, 20], color=COLOR_IEC, stroke_width=2)
            all_curves.add(c)
            
        # IEEE (Purple)
        for i, (name, A, B, p) in enumerate(IEEE_PARAMS):
            c = axes.plot(lambda m, A=A, B=B, p=p: min(10, max(0, 1.0*((A/(m**p - 1)) + B))), x_range=[1.05, 20], color=COLOR_IEEE, stroke_width=2)
            all_curves.add(c)

        self.play(Create(all_curves), run_time=4)

        # Legenda Final
        legenda = VGroup(
            Text("IEC (Azul)", color=COLOR_IEC, font_size=24),
            Text("IEEE (Rox)", color=COLOR_IEEE, font_size=24)
        ).arrange(RIGHT, buff=1).to_edge(UP, buff=1)
        
        leg_bg = SurroundingRectangle(legenda, color=WHITE, fill_color=BLACK, fill_opacity=0.8)
        
        self.play(FadeIn(leg_bg), FadeIn(legenda))
        
        # --- CORREÇÃO 3: Remove o FadeOut final e mantém a tela ---
        self.wait(5)


class EqTCCNum_Tese(Scene):
    def construct(self):
        # ==========================================
        # 1. TÍTULO
        # ==========================================
        title = Text("Curvas TCC: Da Teoria à Prática", font_size=36, weight=BOLD)
        title.to_edge(UP, buff=0.5)
        underline = Line(LEFT, RIGHT, color=BLUE).match_width(title).next_to(title, DOWN, buff=0.1)
        
        self.play(Write(title), Create(underline))
        self.wait(0.5)

        # ==========================================
        # 2. FORMA GERAL (LADO ESQUERDO)
        # ==========================================
        # Passamos a string INTEIRA para evitar erros de LaTeX
        eq_general = MathTex(
            r"t(I) = TD \cdot \left( \frac{A}{\left(\frac{I}{I_p}\right)^p - 1} + B \right)"
        ).scale(0.8).shift(UP*0.5 + LEFT*3)

        # Colorindo de forma inteligente (busca o texto dentro da fórmula)
        eq_general.set_color_by_tex("t(I)", PINK)
        eq_general.set_color_by_tex("TD", BLUE)
        eq_general.set_color_by_tex("A", TEAL)
        eq_general.set_color_by_tex("B", TEAL)
        eq_general.set_color_by_tex("p", TEAL)
        # O "I" aparece duas vezes, set_color_by_tex colore ambas
        eq_general.set_color_by_tex("I", PINK) 
        # O "I_p" contém "I", então garantimos que ele fique amarelo também
        eq_general.set_color_by_tex("I_p", PINK)

        label_general = Text("Forma Geral (IEC/IEEE)", font_size=24, color=GRAY_A)
        label_general.next_to(eq_general, UP, aligned_edge=LEFT)

        self.play(FadeIn(label_general), Write(eq_general))
        self.wait(1)

        # ==========================================
        # 3. EXEMPLO NUMÉRICO (LADO DIREITO)
        # ==========================================
        label_ex = Text("Exemplo: IEC Standard Inverse", font_size=24, color=GRAY_A)
        label_ex.shift(UP*0.5 + RIGHT*3)
        
        params = VGroup(
            MathTex(r"A = 0.14, \quad B = 0, \quad p = 0.02", color=TEAL),
            MathTex(r"TD = 0.5", color=BLUE),
            MathTex(r"I = 500\text{A}, \quad I_p = 100\text{A}", color=PINK)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).scale(0.6)
        params.next_to(label_ex, DOWN, aligned_edge=LEFT)

        self.play(FadeIn(label_ex), Write(params))
        self.wait(1)

        # EQUAÇÃO NUMÉRICA (Onde estava o erro)
        # String única para não quebrar o LaTeX
        eq_numeric = MathTex(
            r"t = 0.5 \cdot \left( \frac{0.14}{\left(\frac{500}{100}\right)^{0.02} - 1} + 0 \right)"
        ).scale(0.6).next_to(params, DOWN, buff=0.3, aligned_edge=LEFT)
        
        # --- COLORINDO SEM USAR ÍNDICES MANUAIS PERIGOSOS ---
        
        # 1. Variável t
        eq_numeric.set_color_by_tex("t", PINK)
        
        # 2. Constantes e Valores (Busca pela string exata)
        eq_numeric.set_color_by_tex("0.5", BLUE)   # TD
        eq_numeric.set_color_by_tex("0.14", TEAL)  # A
        eq_numeric.set_color_by_tex("0.02", TEAL)  # p
        eq_numeric.set_color_by_tex("500", PINK) # I
        eq_numeric.set_color_by_tex("100", PINK) # Ip

        # 3. O termo B (+ 0)
        # Como "0" existe em vários lugares (0.5, 100...), usamos índice NEGATIVO
        # [-1] é o parêntese de fechamento ")"
        # [-2] é o número "0" final
        eq_numeric[0][-2].set_color(TEAL)

        self.play(Write(eq_numeric))
        self.wait(2)

        # ==========================================
        # 4. CONCLUSÃO
        # ==========================================
        final_box = SurroundingRectangle(eq_numeric, color=PINK, buff=0.2)
        result_text = MathTex(r"\approx 2.97\text{ s}", color=PINK).next_to(eq_numeric, RIGHT)

        self.play(Create(final_box))
        self.play(Write(result_text))
        self.wait(2)

        # Limpeza Final
        self.play(
            FadeOut(params), FadeOut(label_ex), FadeOut(final_box), FadeOut(result_text),
            FadeOut(eq_numeric), FadeOut(label_general),
            eq_general.animate.move_to(ORIGIN).scale(1.2)
        )
        
        final_note = Text("Essa estrutura modela todas as curvas por meio de modelagem matemática.", font_size=20, color=GRAY)
        final_note.next_to(eq_general, DOWN, buff=0.5)
        self.play(FadeIn(final_note))
        self.wait(2)