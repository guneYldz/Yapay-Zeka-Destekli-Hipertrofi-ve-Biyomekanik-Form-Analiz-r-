from data.database import verify_user, register_user, update_user_stats, get_all_users, delete_user

class UserService:
    @staticmethod
    def login(username, password):
        # UI ile Database arasındaki iş mantığı köprüsü
        user = verify_user(username, password)
        if user:
            return user
        return None

    @staticmethod
    def register(username, password):
        if not username or not password:
            return False
        return register_user(username, password)

    @staticmethod
    def update_stats(user_id, height, weight):
        if height > 0 and weight > 0:
            return update_user_stats(user_id, height, weight)
        return False

    @staticmethod
    def list_all_users():
        return get_all_users()

    @staticmethod
    def remove_user(user_id):
        return delete_user(user_id)
