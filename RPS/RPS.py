class RPS:
    """
    가위바위보 게임을 관리하는 클래스
    """
    
    # 게임 상수 정의
    ROCK = 0
    PAPER = 1
    SCISSORS = 2
    
    # 게임 결과 상수
    WIN = 1
    LOSE = -1
    DRAW = 0
    
    def __init__(self):
        self.choices = ['가위', '바위', '보']
        self.choice_symbols = ['✂️', '✊', '✋']
    
    def get_choice_name(self, choice):
        """선택 번호를 한글 이름으로 변환"""
        if 0 <= choice <= 2:
            return self.choices[choice]
        return "잘못된 선택"
    
    def get_choice_symbol(self, choice):
        """선택 번호를 이모지로 변환"""
        if 0 <= choice <= 2:
            return self.choice_symbols[choice]
        return "❓"
    
    def determine_winner(self, player1_choice, player2_choice):
        """
        두 플레이어의 선택을 비교하여 승부 결과를 결정
        
        Args:
            player1_choice (int): 플레이어1의 선택 (0: 가위, 1: 바위, 2: 보)
            player2_choice (int): 플레이어2의 선택 (0: 가위, 1: 바위, 2: 보)
        
        Returns:
            tuple: (결과, 승자)
            - 결과: 1(플레이어1 승리), -1(플레이어2 승리), 0(무승부)
            - 승자: "player1", "player2", "draw"
        """
        if not (0 <= player1_choice <= 2 and 0 <= player2_choice <= 2):
            raise ValueError("선택은 0(가위), 1(바위), 2(보) 중 하나여야 합니다.")
        
        if player1_choice == player2_choice:
            return self.DRAW, "draw"
        
        # 가위바위보 규칙: 가위(0) < 바위(1) < 보(2) < 가위(0)
        if (player1_choice + 1) % 3 == player2_choice:
            return self.LOSE, "player2"
        else:
            return self.WIN, "player1"
    
    def play_round(self, player1_choice, player2_choice):
        """
        한 라운드의 게임을 진행하고 결과를 반환
        
        Args:
            player1_choice (int): 플레이어1의 선택
            player2_choice (int): 플레이어2의 선택
        
        Returns:
            dict: 게임 결과 정보
        """
        result, winner = self.determine_winner(player1_choice, player2_choice)
        
        return {
            'player1_choice': player1_choice,
            'player2_choice': player2_choice,
            'player1_choice_name': self.get_choice_name(player1_choice),
            'player2_choice_name': self.get_choice_name(player2_choice),
            'player1_choice_symbol': self.get_choice_symbol(player1_choice),
            'player2_choice_symbol': self.get_choice_symbol(player2_choice),
            'result': result,
            'winner': winner
        }
    
    def print_round_result(self, game_result):
        """게임 결과를 예쁘게 출력"""
        print(f"\n=== 가위바위보 결과 ===")
        print(f"플레이어1: {game_result['player1_choice_symbol']} {game_result['player1_choice_name']}")
        print(f"플레이어2: {game_result['player2_choice_symbol']} {game_result['player2_choice_name']}")
        
        if game_result['winner'] == "draw":
            print("결과: 무승부! 🤝")
        elif game_result['winner'] == "player1":
            print("결과: 플레이어1 승리! 🎉")
        else:
            print("결과: 플레이어2 승리! 🎉")
        print("=" * 20)
