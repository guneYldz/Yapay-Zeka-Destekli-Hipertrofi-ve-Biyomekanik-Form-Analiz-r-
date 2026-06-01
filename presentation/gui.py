import customtkinter as ctk
from tkinter import messagebox
import sys
import os
from PIL import Image, ImageEnhance

from data.database import register_user, verify_user, init_db, update_user_stats, get_all_users, delete_user

# Koyu tema ve neon renk ayarları
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

NEON_CYAN = "#00F3FF"
NEON_CYAN_HOVER = "#00C2CC"
NEON_PINK = "#FF00AA"
NEON_PINK_HOVER = "#CC0088"
BG_DARK = "#0D0D12"
CARD_DARK = "#1A1A24"


class FormAnalyzerApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("Biyomekanik Form Analizörü")
        self.geometry("800x500")
        self.resizable(False, False)
        self.configure(fg_color=BG_DARK)
        
        # Ekranı ortala
        self.eval('tk::PlaceWindow . center')
        
        # Veritabanını hazırla
        init_db()

        # Arka plan resmi yükleme ve soluklaştırma (faded)
        bg_path = "bg.png"
        if os.path.exists(bg_path):
            try:
                pil_img = Image.open(bg_path).convert("RGBA")
                # Resmi silikleştirmek için alfa kanalını ve parlaklığı düşürüyoruz
                enhancer = ImageEnhance.Brightness(pil_img)
                pil_img = enhancer.enhance(0.2)  # %20 parlaklık (karanlık ve silik)
                
                bg_image = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(800, 500))
                self.bg_label = ctk.CTkLabel(self, image=bg_image, text="")
                self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)
            except Exception as e:
                print(f"Arka plan resmi yüklenemedi: {e}")

        # Sol Panel (Banner)
        self.sidebar_frame = ctk.CTkFrame(self, width=300, corner_radius=0, fg_color="#101016") # Yarı şeffaflık hissi için koyu
        self.sidebar_frame.pack(side="left", fill="y")
        self.sidebar_frame.pack_propagate(False)
        
        self.logo_label = ctk.CTkLabel(
            self.sidebar_frame, 
            text="AI FORM", 
            font=ctk.CTkFont(size=40, weight="bold"),
            text_color=NEON_CYAN
        )
        self.logo_label.pack(pady=(150, 5))
        
        self.sub_logo = ctk.CTkLabel(
            self.sidebar_frame, 
            text="ANALYSIS PRO", 
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=NEON_PINK
        )
        self.sub_logo.pack(pady=0)

        # Sağ Panel (Ana İçerik)
        self.main_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.main_frame.pack(side="right", fill="both", expand=True)

        # View state
        self.current_view = None
        self.show_login_view()

    def show_login_view(self):
        if self.current_view:
            self.current_view.destroy()
            
        self.current_view = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.current_view.pack(expand=True, fill="both", padx=60, pady=60)

        title = ctk.CTkLabel(self.current_view, text="GİRİŞ YAP", font=ctk.CTkFont(size=28, weight="bold"), text_color="white")
        title.pack(pady=(0, 40))

        self.entry_username_l = ctk.CTkEntry(self.current_view, placeholder_text="Kullanıcı Adı", width=300, height=45, fg_color="#181820", border_color="#333", border_width=1)
        self.entry_username_l.pack(pady=10)

        self.entry_password_l = ctk.CTkEntry(self.current_view, placeholder_text="Şifre", show="*", width=300, height=45, fg_color="#181820", border_color="#333", border_width=1)
        self.entry_password_l.pack(pady=10)

        btn_login = ctk.CTkButton(self.current_view, text="GİRİŞ", command=self.login_event, width=300, height=45, fg_color=NEON_CYAN, hover_color=NEON_CYAN_HOVER, text_color="black", font=ctk.CTkFont(weight="bold"))
        btn_login.pack(pady=(30, 10))

        btn_switch = ctk.CTkButton(self.current_view, text="Hesabın yok mu? Kayıt Ol", command=self.show_register_view, width=300, height=35, fg_color="transparent", hover_color=CARD_DARK, text_color=NEON_PINK)
        btn_switch.pack()

    def show_register_view(self):
        if self.current_view:
            self.current_view.destroy()
            
        self.current_view = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.current_view.pack(expand=True, fill="both", padx=60, pady=60)

        title = ctk.CTkLabel(self.current_view, text="YENİ HESAP", font=ctk.CTkFont(size=28, weight="bold"), text_color="white")
        title.pack(pady=(0, 40))

        self.entry_username_r = ctk.CTkEntry(self.current_view, placeholder_text="Yeni Kullanıcı Adı", width=300, height=45, fg_color="#181820", border_color="#333", border_width=1)
        self.entry_username_r.pack(pady=10)

        self.entry_password_r = ctk.CTkEntry(self.current_view, placeholder_text="Yeni Şifre", show="*", width=300, height=45, fg_color="#181820", border_color="#333", border_width=1)
        self.entry_password_r.pack(pady=10)

        btn_register = ctk.CTkButton(self.current_view, text="KAYIT OL", command=self.register_event, width=300, height=45, fg_color=NEON_PINK, hover_color=NEON_PINK_HOVER, text_color="white", font=ctk.CTkFont(weight="bold"))
        btn_register.pack(pady=(30, 10))

        btn_switch = ctk.CTkButton(self.current_view, text="Zaten hesabın var mı? Giriş Yap", command=self.show_login_view, width=300, height=35, fg_color="transparent", hover_color=CARD_DARK, text_color=NEON_CYAN)
        btn_switch.pack()

    def login_event(self):
        username = self.entry_username_l.get().strip()
        password = self.entry_password_l.get().strip()
        if not username or not password:
            messagebox.showwarning("Uyarı", "Lütfen tüm alanları doldurun.")
            return
            
        user = verify_user(username, password)
        if user:
            self.logged_in_user = user
            self.show_dashboard_view()
        else:
            messagebox.showerror("Hata", "Kullanıcı adı veya şifre hatalı!")

    def show_dashboard_view(self):
        if self.current_view:
            self.current_view.destroy()
            
        self.current_view = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.current_view.pack(expand=True, fill="both", padx=40, pady=40)

        # Hoşgeldin ve Rol
        role_color = NEON_CYAN if self.logged_in_user['role'] != 'admin' else NEON_PINK
        welcome_lbl = ctk.CTkLabel(self.current_view, text=f"Hoşgeldin, {self.logged_in_user['username']}!", font=ctk.CTkFont(size=24, weight="bold"), text_color="white")
        welcome_lbl.pack(pady=(0, 5))
        
        role_lbl = ctk.CTkLabel(self.current_view, text=f"Yetki: {self.logged_in_user['role'].upper()}", font=ctk.CTkFont(size=14), text_color=role_color)
        role_lbl.pack(pady=(0, 20))

        # Boy / Kilo Alanı
        metrics_frame = ctk.CTkFrame(self.current_view, fg_color=CARD_DARK, corner_radius=10)
        metrics_frame.pack(fill="x", pady=10, padx=20)
        
        ctk.CTkLabel(metrics_frame, text="Vücut Kitle İndeksi (VKİ) Hesaplama", font=ctk.CTkFont(size=16, weight="bold"), text_color="white").pack(pady=(10,5))
        
        input_frame = ctk.CTkFrame(metrics_frame, fg_color="transparent")
        input_frame.pack(pady=5)
        
        self.entry_height = ctk.CTkEntry(input_frame, placeholder_text="Boy (cm) örn: 180", width=140)
        self.entry_height.pack(side="left", padx=10)
        if self.logged_in_user['height']: self.entry_height.insert(0, str(int(self.logged_in_user['height'])))
        
        self.entry_weight = ctk.CTkEntry(input_frame, placeholder_text="Kilo (kg) örn: 75", width=140)
        self.entry_weight.pack(side="left", padx=10)
        if self.logged_in_user['weight']: self.entry_weight.insert(0, str(int(self.logged_in_user['weight'])))

        btn_calc = ctk.CTkButton(metrics_frame, text="Hesapla ve Kaydet", command=self.calculate_bmi, width=150, fg_color="#333", hover_color="#444")
        btn_calc.pack(pady=10)
        
        self.lbl_bmi_result = ctk.CTkLabel(metrics_frame, text="", font=ctk.CTkFont(size=14))
        self.lbl_bmi_result.pack(pady=(0, 10))

        # Başlat Butonu
        btn_start = ctk.CTkButton(self.current_view, text="🚀 ANTRENMANI BAŞLAT", command=self.start_analyzer, width=300, height=60, fg_color=NEON_CYAN, hover_color=NEON_CYAN_HOVER, text_color="black", font=ctk.CTkFont(size=18, weight="bold"))
        btn_start.pack(pady=30)

        # Admin Paneli Butonu
        if self.logged_in_user['role'] == 'admin':
            btn_admin = ctk.CTkButton(self.current_view, text="⚙️ Yönetim Paneli", command=self.show_admin_view, width=200, height=35, fg_color=NEON_PINK, hover_color=NEON_PINK_HOVER, text_color="white")
            btn_admin.pack(pady=10)

    def calculate_bmi(self):
        try:
            h = float(self.entry_height.get())
            w = float(self.entry_weight.get())
            
            # Kaydet
            update_user_stats(self.logged_in_user['id'], h, w)
            self.logged_in_user['height'] = h
            self.logged_in_user['weight'] = w
            
            bmi = w / ((h/100)**2)
            
            if bmi < 18.5:
                res = "Zayıf - Kas Kütlesi Eklemelisin!"
                col = NEON_CYAN
            elif 18.5 <= bmi < 24.9:
                res = "İdeal - Formunu Koru!"
                col = "#00FF00"
            elif 25 <= bmi < 29.9:
                res = "Fazla Kilolu - Kardiyo Ekleyebilirsin!"
                col = "orange"
            else:
                res = "Obezite - Dikkatli Antrenman!"
                col = NEON_PINK
                
            self.lbl_bmi_result.configure(text=f"VKİ: {bmi:.1f} | Tavsiye: {res}", text_color=col)
        except ValueError:
            messagebox.showerror("Hata", "Lütfen geçerli sayılar girin.")

    def show_admin_view(self):
        if self.current_view:
            self.current_view.destroy()
            
        self.current_view = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.current_view.pack(expand=True, fill="both", padx=40, pady=40)

        title = ctk.CTkLabel(self.current_view, text="⚙️ YÖNETİM PANELİ", font=ctk.CTkFont(size=24, weight="bold"), text_color=NEON_PINK)
        title.pack(pady=(0, 20))

        # Kullanıcı Listesi
        list_frame = ctk.CTkScrollableFrame(self.current_view, width=400, height=250, fg_color=CARD_DARK)
        list_frame.pack(pady=10)

        users = get_all_users()
        for u in users:
            row = ctk.CTkFrame(list_frame, fg_color="transparent")
            row.pack(fill="x", pady=5)
            
            lbl = ctk.CTkLabel(row, text=f"{u['username']} ({u['role']})", text_color="white", width=250, anchor="w")
            lbl.pack(side="left", padx=10)
            
            if u['id'] != self.logged_in_user['id']: # Kendini silemesin
                btn_del = ctk.CTkButton(row, text="Sil", width=60, fg_color="red", hover_color="#aa0000", command=lambda uid=u['id']: self.delete_user_event(uid))
                btn_del.pack(side="right", padx=10)

        btn_back = ctk.CTkButton(self.current_view, text="Geri Dön", command=self.show_dashboard_view, width=200, fg_color="#333", hover_color="#444")
        btn_back.pack(pady=20)

    def delete_user_event(self, uid):
        if messagebox.askyesno("Onay", "Kullanıcıyı silmek istediğinize emin misiniz?"):
            delete_user(uid)
            self.show_admin_view() # Yenile

    def register_event(self):
        username = self.entry_username_r.get().strip()
        password = self.entry_password_r.get().strip()
        if not username or not password:
            messagebox.showwarning("Uyarı", "Lütfen tüm alanları doldurun.")
            return
        success = register_user(username, password)
        if success:
            messagebox.showinfo("Başarılı", "Kayıt başarılı! Lütfen giriş yapın.")
            self.show_login_view()
        else:
            messagebox.showerror("Hata", "Bu kullanıcı adı alınmış!")

    def start_analyzer(self):
        self.destroy()
        from form_analyzer import main
        try:
            main()
        except Exception as e:
            print(f"Kamera başlatılırken hata oluştu: {e}")
            sys.exit(1)

def run_gui():
    app = FormAnalyzerApp()
    app.mainloop()

if __name__ == "__main__":
    run_gui()
