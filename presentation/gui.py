import flet as ft
import os
import sys
import time

# Proje kök dizinini path'e ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from application.user_service import UserService
from data.database import init_db

# --- MODERN TASARIM SİSTEMİ ---
PRIMARY = "#10B981"
PRIMARY_LIGHT = "#34D399"
BG_DARK = "#0F172A"
CARD_BG = "#1E293B"
TEXT_MAIN = "#F8FAFC"
TEXT_DIM = "#94A3B8"
ACCENT = "#F59E0B"

class FormAnalyzerApp:
    def __init__(self, page: ft.Page):
        self.page = page
        self.page.title = "AI Biomechanical Form Analyzer"
        self.page.theme_mode = "dark"
        self.page.bgcolor = BG_DARK
        self.page.window_width = 1100
        self.page.window_height = 800
        self.page.window_resizable = False
        
        # Font Yükleme
        self.page.fonts = {
            "Outfit": "https://github.com/google/fonts/raw/main/ofl/outfit/Outfit-VariableFont_wght.ttf"
        }
        self.page.theme = ft.Theme(font_family="Outfit")
        
        self.user_data = None
        init_db()
        self.show_login()

    def clear_page(self):
        self.page.controls.clear()
        self.page.update()

    def show_login(self, e=None):
        self.clear_page()
        
        username_field = ft.TextField(
            label="Kullanıcı Adı",
            icon="person",
            border_color=PRIMARY,
            focused_border_color=PRIMARY_LIGHT,
            width=350,
            border_radius=12,
        )
        password_field = ft.TextField(
            label="Şifre",
            icon="lock",
            password=True,
            can_reveal_password=True,
            border_color=PRIMARY,
            focused_border_color=PRIMARY_LIGHT,
            width=350,
            border_radius=12,
        )

        login_card = ft.Container(
            content=ft.Column(
                [
                    ft.Text("AI FORM", size=48, weight="bold", color=PRIMARY),
                    ft.Text("Biomechanical Analysis Pro", color=TEXT_DIM, size=16),
                    ft.Divider(height=40, color="transparent"),
                    username_field,
                    password_field,
                    ft.Divider(height=20, color="transparent"),
                    ft.ElevatedButton(
                        "GİRİŞ YAP",
                        width=350,
                        height=55,
                        style=ft.ButtonStyle(
                            bgcolor=PRIMARY,
                            color=BG_DARK,
                            shape=ft.RoundedRectangleBorder(radius=12),
                            elevation=10,
                        ),
                        on_click=lambda _: self.handle_login(username_field.value, password_field.value),
                    ),
                    ft.TextButton(
                        "Hesabın yok mu? Yeni bir profil oluştur",
                        on_click=self.show_register,
                        style=ft.ButtonStyle(color=PRIMARY_LIGHT),
                    ),
                ],
                horizontal_alignment="center",
            ),
            padding=50,
            bgcolor=CARD_BG,
            border_radius=24,
            shadow=ft.BoxShadow(blur_radius=40, color="#00000066", spread_radius=-10),
        )

        self.page.add(
            ft.Container(
                content=login_card,
                alignment=ft.Alignment(0, 0),
                expand=True,
                bgcolor=BG_DARK
            )
        )

    def show_register(self, e=None):
        self.clear_page()
        
        username_field = ft.TextField(label="Yeni Kullanıcı Adı", icon="person_add", width=350, border_radius=12)
        password_field = ft.TextField(label="Yeni Şifre", icon="lock_open", password=True, can_reveal_password=True, width=350, border_radius=12)

        register_card = ft.Container(
            content=ft.Column(
                [
                    ft.Text("YENİ KAYIT", size=32, weight="bold", color=ACCENT),
                    ft.Text("Performans yolculuğuna bugün katıl", color=TEXT_DIM),
                    ft.Divider(height=40, color="transparent"),
                    username_field,
                    password_field,
                    ft.Divider(height=20, color="transparent"),
                    ft.ElevatedButton(
                        "HESAP OLUŞTUR",
                        width=350,
                        height=55,
                        style=ft.ButtonStyle(bgcolor=ACCENT, color=BG_DARK, shape=ft.RoundedRectangleBorder(radius=12)),
                        on_click=lambda _: self.handle_register(username_field.value, password_field.value),
                    ),
                    ft.TextButton("Zaten hesabın var mı? Giriş yap", on_click=self.show_login),
                ],
                horizontal_alignment="center",
            ),
            padding=50,
            bgcolor=CARD_BG,
            border_radius=24,
        )

        self.page.add(ft.Container(content=register_card, alignment=ft.Alignment(0, 0), expand=True, bgcolor=BG_DARK))

    def handle_login(self, username, password):
        if not username or not password:
            self.show_snack("Bütün alanları doldurmalısın.")
            return
        user = UserService.login(username, password)
        if user:
            self.user_data = user
            self.show_dashboard()
        else:
            self.show_snack("Hatalı giriş bilgileri.")

    def handle_register(self, username, password):
        if not username or not password: return
        if UserService.register(username, password):
            self.show_snack("Başarıyla kayıt oldun! Giriş yapabilirsin.")
            self.show_login()
        else:
            self.show_snack("Bu kullanıcı adı sistemde mevcut.")

    def show_snack(self, message):
        self.page.snack_bar = ft.SnackBar(ft.Text(message), bgcolor=CARD_BG)
        self.page.snack_bar.open = True
        self.page.update()

    def show_dashboard(self):
        self.clear_page()
        
        # Header
        header = ft.Container(
            content=ft.Row(
                [
                    ft.Column([
                        ft.Text(f"Hoş Geldin, {self.user_data['username']}", size=32, weight="bold", color=TEXT_MAIN),
                        ft.Row([
                            ft.Icon("verified", size=18, color=PRIMARY_LIGHT),
                            ft.Text(f"{self.user_data['role'].upper()} SEVİYESİ", size=12, color=PRIMARY_LIGHT, weight="bold"),
                        ]),
                    ], spacing=2),
                    ft.IconButton(icon="logout", icon_color=TEXT_DIM, on_click=self.show_login, tooltip="Oturumu Kapat"),
                ],
                alignment="spaceBetween",
            ),
        )

        # Space after header
        header_spacer = ft.Divider(height=20, color="transparent")

        # BMI Logic
        bmi_val = 0
        bmi_lbl, bmi_clr = "Veri Bekleniyor", TEXT_DIM
        if self.user_data['height'] and self.user_data['weight']:
            h, w = self.user_data['height']/100, self.user_data['weight']
            bmi_val = w / (h*h)
            if bmi_val < 18.5: bmi_lbl, bmi_clr = "Zayıf", "blue400"
            elif 18.5 <= bmi_val < 25: bmi_lbl, bmi_clr = "Normal", PRIMARY
            elif 25 <= bmi_val < 30: bmi_lbl, bmi_clr = "Kilolu", "orange400"
            else: bmi_lbl, bmi_clr = "Obez", "red400"

        # Health Card
        self.height_input = ft.TextField(value=str(int(self.user_data['height'])) if self.user_data['height'] else "", width=80, height=45, text_size=14, border_radius=8)
        self.weight_input = ft.TextField(value=str(int(self.user_data['weight'])) if self.user_data['weight'] else "", width=80, height=45, text_size=14, border_radius=8)

        health_card = ft.Container(
            content=ft.Column([
                ft.Text("Vücut Durumu", size=20, weight="bold"),
                ft.Row([
                    ft.Column([ft.Text("Boy", size=12, color=TEXT_DIM), self.height_input]),
                    ft.Column([ft.Text("Kilo", size=12, color=TEXT_DIM), self.weight_input]),
                ], spacing=20),
                ft.ElevatedButton("Güncelle", icon="refresh", on_click=self.handle_stats_update, style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8))),
                ft.Container(
                    content=ft.Column([
                        ft.Row([ft.Text("VKİ Endeksi", size=14), ft.Text(f"{bmi_val:.1f}", weight="bold", color=bmi_clr)], alignment="spaceBetween"),
                        ft.ProgressBar(value=min(bmi_val/40, 1), color=bmi_clr, height=10, border_radius=5),
                        ft.Text(bmi_lbl, size=12, color=bmi_clr),
                    ], spacing=10),
                    margin=10
                )
            ], spacing=20),
            padding=30, bgcolor=CARD_BG, border_radius=20, expand=2 # INTEGER EXPAND
        )

        # Action Card
        action_card = ft.Container(
            content=ft.Column([
                # Sadece modern dambıl ikonu kalsın
                ft.Container(content=ft.Icon("fitness_center", size=60, color=PRIMARY), padding=20, bgcolor="#10B9811A", border_radius=30),
                
                # Orijinal metinler
                ft.Text("Form Analizi", size=26, weight="bold", color=TEXT_MAIN),
                ft.Text("Biyomekanik formunu analiz etmek için kameranı başlat.", color=TEXT_DIM, text_align="center"),
                
                ft.Divider(height=30, color="transparent"),
                
                # Buton
                ft.Container(
                    content=ft.ElevatedButton(
                        "ANTRENMANI BAŞLAT", icon="play_circle", width=300, height=70,
                        style=ft.ButtonStyle(bgcolor=PRIMARY, color=BG_DARK, shape=ft.RoundedRectangleBorder(radius=15), elevation=15),
                        on_click=self.start_analyzer_flow
                    ),
                    shadow=ft.BoxShadow(blur_radius=30, spread_radius=-10, color=PRIMARY)
                ),
            ], horizontal_alignment="center", spacing=15, alignment="center"),
            padding=30, bgcolor=CARD_BG, border_radius=25, expand=3
        )

        # Admin path
        admin_path = ft.Container()
        if self.user_data['role'] == 'admin':
            admin_path = ft.Container(
                content=ft.TextButton("Yönetim Paneli", icon="admin_panel_settings", on_click=self.show_admin),
                margin=20
            )

        # Layout
        self.page.add(
            ft.Container(
                content=ft.Column([
                    header,
                    header_spacer,
                    ft.Row([health_card, action_card], spacing=30, expand=True),
                    admin_path
                ], expand=True),
                padding=40, 
                expand=True,
                bgcolor=BG_DARK
            )
        )

    def handle_stats_update(self, e):
        try:
            h, w = float(self.height_input.value), float(self.weight_input.value)
            UserService.update_stats(self.user_data['id'], h, w)
            self.user_data['height'], self.user_data['weight'] = h, w
            self.show_dashboard()
            self.show_snack("Veriler başarıyla güncellendi.")
        except: self.show_snack("Lütfen geçerli sayılar girin.")

    def show_admin(self, e):
        self.clear_page()
        users = UserService.list_all_users()
        rows = [ft.DataRow(cells=[
            ft.DataCell(ft.Text(str(u['id']))),
            ft.DataCell(ft.Text(u['username'])),
            ft.DataCell(ft.Text(u['role'])),
            ft.DataCell(ft.IconButton("delete", icon_color="red", on_click=lambda _, uid=u['id']: self.handle_delete_user(uid), visible=(u['id']!=self.user_data['id']))),
        ]) for u in users]

        self.page.add(ft.Container(
            content=ft.Column([
                ft.Row([ft.Text("Yönetim Paneli", size=32, weight="bold"), ft.ElevatedButton("Geri Dön", icon="arrow_back", on_click=lambda _: self.show_dashboard())], alignment="spaceBetween"),
                ft.DataTable(columns=[ft.DataColumn(ft.Text("ID")), ft.DataColumn(ft.Text("Kullanıcı")), ft.DataColumn(ft.Text("Rol")), ft.DataColumn(ft.Text("İşlem"))], rows=rows, bgcolor=CARD_BG, border_radius=15, expand=True)
            ]), padding=40, expand=True
        ))

    def handle_delete_user(self, uid):
        if UserService.remove_user(uid): self.show_admin(None)

    def start_analyzer_flow(self, e):
        self.page.window_close()
        time.sleep(0.5)
        import form_analyzer
        print("\n[INFO] CV Analizörü başlatılıyor...")
        try: form_analyzer.main()
        except Exception as ex: print(f"[ERROR] Hata: {ex}")

def run_gui():
    ft.app(target=FormAnalyzerApp)

if __name__ == "__main__":
    run_gui()
