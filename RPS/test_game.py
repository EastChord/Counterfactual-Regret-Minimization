#!/usr/bin/env python3
"""
가위바위보 게임 테스트 스크립트
"""

from RPS import RPS
from RPSPlayer import RPSPlayer, RPSGame

def test_basic_game_logic():
    """기본 게임 로직 테스트"""
    print("=== 기본 게임 로직 테스트 ===")
    
    game = RPS()
    
    # 테스트 케이스들
    test_cases = [
        (0, 0, "무승부"),  # 가위 vs 가위
        (0, 1, "플레이어2 승리"),  # 가위 vs 바위
        (0, 2, "플레이어1 승리"),  # 가위 vs 보
        (1, 0, "플레이어1 승리"),  # 바위 vs 가위
        (1, 1, "무승부"),  # 바위 vs 바위
        (1, 2, "플레이어2 승리"),  # 바위 vs 보
        (2, 0, "플레이어2 승리"),  # 보 vs 가위
        (2, 1, "플레이어1 승리"),  # 보 vs 바위
        (2, 2, "무승부"),  # 보 vs 보
    ]
    
    for p1_choice, p2_choice, expected in test_cases:
        result = game.play_round(p1_choice, p2_choice)
        p1_name = game.get_choice_name(p1_choice)
        p2_name = game.get_choice_name(p2_choice)
        
        print(f"{p1_name} vs {p2_name}: {result['winner']} (예상: {expected})")
        
        # 결과 검증
        if expected == "무승부" and result['winner'] != "draw":
            print(f"  ❌ 오류: 무승부가 아닙니다!")
        elif expected == "플레이어1 승리" and result['winner'] != "player1":
            print(f"  ❌ 오류: 플레이어1이 승리하지 않았습니다!")
        elif expected == "플레이어2 승리" and result['winner'] != "player2":
            print(f"  ❌ 오류: 플레이어2가 승리하지 않았습니다!")
        else:
            print(f"  ✅ 정상")

def test_computer_vs_computer():
    """컴퓨터 vs 컴퓨터 게임 테스트"""
    print("\n=== 컴퓨터 vs 컴퓨터 게임 테스트 ===")
    
    game = RPSGame()
    
    # 컴퓨터 플레이어 2명 추가
    player1 = game.add_player("컴퓨터1", is_human=False)
    player2 = game.add_player("컴퓨터2", is_human=False)
    
    print("5라운드 컴퓨터 vs 컴퓨터 게임을 진행합니다...")
    
    for round_num in range(1, 6):
        print(f"\n--- 라운드 {round_num} ---")
        game.play_single_round()
    
    # 최종 결과 출력
    game.print_final_results()

def test_player_stats():
    """플레이어 통계 테스트"""
    print("\n=== 플레이어 통계 테스트 ===")
    
    player = RPSPlayer("테스트플레이어", is_human=False)
    
    # 가상의 게임 결과로 통계 업데이트
    player.update_stats(1)   # 승리
    player.update_stats(-1)  # 패배
    player.update_stats(0)   # 무승부
    player.update_stats(1)   # 승리
    
    player.print_stats()

if __name__ == "__main__":
    print("🎮 가위바위보 게임 테스트를 시작합니다!\n")
    
    # 기본 게임 로직 테스트
    test_basic_game_logic()
    
    # 컴퓨터 vs 컴퓨터 게임 테스트
    test_computer_vs_computer()
    
    # 플레이어 통계 테스트
    test_player_stats()
    
    print("\n🎉 모든 테스트가 완료되었습니다!")
