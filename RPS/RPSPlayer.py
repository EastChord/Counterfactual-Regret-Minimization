import random
from RPS import RPS

class RPSPlayer:
    """
    가위바위보 게임을 하는 플레이어 클래스
    """
    
    def __init__(self, name="플레이어", is_human=True):
        """
        플레이어 초기화
        
        Args:
            name (str): 플레이어 이름
            is_human (bool): 사람 플레이어인지 여부 (True: 사람, False: 컴퓨터)
        """
        self.name = name
        self.is_human = is_human
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.game = RPS()
    
    def get_choice(self):
        """
        플레이어의 선택을 받아옴
        
        Returns:
            int: 선택 (0: 가위, 1: 바위, 2: 보)
        """
        if self.is_human:
            return self._get_human_choice()
        else:
            return self._get_computer_choice()
    
    def _get_human_choice(self):
        """사람 플레이어의 선택을 입력받음"""
        while True:
            try:
                print(f"\n{self.name}님의 차례입니다!")
                print("0: 가위 ✂️")
                print("1: 바위 ✊")
                print("2: 보 ✋")
                
                choice = int(input("선택하세요 (0-2): "))
                
                if 0 <= choice <= 2:
                    return choice
                else:
                    print("잘못된 입력입니다. 0, 1, 2 중에서 선택해주세요.")
            except ValueError:
                print("숫자를 입력해주세요.")
    
    def _get_computer_choice(self):
        """컴퓨터 플레이어의 선택을 랜덤하게 생성"""
        choice = random.randint(0, 2)
        print(f"{self.name}이(가) {self.game.get_choice_symbol(choice)} {self.game.get_choice_name(choice)}를 선택했습니다!")
        return choice
    
    def update_stats(self, result):
        """
        게임 결과에 따라 통계 업데이트
        
        Args:
            result (int): 게임 결과 (1: 승리, -1: 패배, 0: 무승부)
        """
        if result == 1:
            self.wins += 1
        elif result == -1:
            self.losses += 1
        else:
            self.draws += 1
    
    def get_stats(self):
        """플레이어의 통계 정보를 반환"""
        total_games = self.wins + self.losses + self.draws
        win_rate = (self.wins / total_games * 100) if total_games > 0 else 0
        
        return {
            'name': self.name,
            'wins': self.wins,
            'losses': self.losses,
            'draws': self.draws,
            'total_games': total_games,
            'win_rate': win_rate
        }
    
    def print_stats(self):
        """플레이어의 통계를 출력"""
        stats = self.get_stats()
        print(f"\n=== {self.name}의 통계 ===")
        print(f"승리: {stats['wins']}회")
        print(f"패배: {stats['losses']}회")
        print(f"무승부: {stats['draws']}회")
        print(f"총 게임: {stats['total_games']}회")
        print(f"승률: {stats['win_rate']:.1f}%")
        print("=" * 20)


class RPSGame:
    """
    가위바위보 게임을 관리하는 메인 클래스
    """
    
    def __init__(self):
        self.game = RPS()
        self.players = []
    
    def add_player(self, name, is_human=True):
        """플레이어를 게임에 추가"""
        player = RPSPlayer(name, is_human)
        self.players.append(player)
        return player
    
    def play_single_round(self):
        """한 라운드의 게임을 진행"""
        if len(self.players) != 2:
            print("플레이어가 2명이어야 합니다!")
            return None
        
        player1, player2 = self.players
        
        # 각 플레이어의 선택 받기
        choice1 = player1.get_choice()
        choice2 = player2.get_choice()
        
        # 게임 결과 계산
        game_result = self.game.play_round(choice1, choice2)
        
        # 결과 출력
        self.game.print_round_result(game_result)
        
        # 플레이어 통계 업데이트
        player1.update_stats(game_result['result'])
        player2.update_stats(-game_result['result'])  # 플레이어2는 반대 결과
        
        return game_result
    
    def play_multiple_rounds(self, rounds=3):
        """여러 라운드의 게임을 진행"""
        print(f"\n🎮 {rounds}라운드 가위바위보 게임을 시작합니다!")
        
        for round_num in range(1, rounds + 1):
            print(f"\n--- 라운드 {round_num} ---")
            self.play_single_round()
        
        # 최종 결과 출력
        self.print_final_results()
    
    def print_final_results(self):
        """최종 게임 결과를 출력"""
        print(f"\n🏆 최종 결과 🏆")
        for player in self.players:
            player.print_stats()
        
        # 승자 결정
        if self.players[0].wins > self.players[1].wins:
            print(f"\n🎉 {self.players[0].name}님이 승리했습니다!")
        elif self.players[1].wins > self.players[0].wins:
            print(f"\n🎉 {self.players[1].name}님이 승리했습니다!")
        else:
            print(f"\n🤝 무승부입니다!")


# 게임 실행 예제
if __name__ == "__main__":
    # 게임 인스턴스 생성
    game = RPSGame()
    
    # 플레이어 추가
    game.add_player("사용자", is_human=True)
    game.add_player("컴퓨터", is_human=False)
    
    # 게임 시작
    game.play_multiple_rounds(3)
