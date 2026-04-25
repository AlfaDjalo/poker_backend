
from app.db.session import engine, SessionLocal
from app.db.base import Base 

from app.db.models.players import Player
from app.db.models.poker_tables import PokerTable

from app.db.models.bankroll_transaction import BankrollTransaction

from app.db.models.poker_sessions import PokerSession
from app.db.models.table_seating import TableSeat

from app.db.models.betting_config import BettingConfig
from app.db.models.betting_config_details import BettingConfigDetails

from app.db.models.hands import Hand
from app.db.models.hand_points import HandPoint

from app.db.models.hole_cards import HoleCard
from app.db.models.board_cards import BoardCard
from app.db.models.actions import Action
from app.db.models.card_events import CardEvent

from app.db.models.point_nodes import PointNode
from app.db.models.point_results import PointResult
from app.db.models.point_cards import PointCard
from app.db.models.payouts import Payout

def seed_player_tables():
    """Create default poker tables if they don't exist."""
    db = SessionLocal()
    deposit_amount = 100

    try:
        table_name = "Table 1"
        # Check if Table 1 already exists
        existing = db.query(PokerTable).filter(
            PokerTable.table_name == "Table 1"
        ).first()
        
        if not existing:
            table = PokerTable(
                table_name="Table 1",
                max_players=6
            )
            db.add(table)
            db.commit()
            print(f"✓ Created {table_name} with max_players=6")
        else:
            print(f"✓ {table_name} already exists")
    
        for i in range(1, 7):
            player_name = f"Player {i}"

            existing_player = db.query(Player).filter(
                Player.username == player_name
            ).first()

            if not existing_player:
                new_player = Player(
                    username=player_name,
                    is_bot=False,
                )
                db.add(new_player)
                db.flush()
                print(f"✓ Created '{player_name}'")

                deposit = BankrollTransaction(
                    player_id=new_player.player_id,
                    transaction_type="DEPOSIT",
                    amount=deposit_amount,
                    hand_id=None,
                    balance_after=deposit_amount,
                )
                db.add(deposit)

            else:
                print(f"✓ '{player_name}' already exists")
            
        db.commit()

    except Exception as e:
        print(f"Error during seeding: {e}")
        db.rollback()

    finally:
        db.close()


def seed_betting_config_tables():
    """Create default poker tables if they don't exist."""
    db = SessionLocal()

    try:
        config_name = "1/2"
        betting_config = db.query(BettingConfig).filter(
            BettingConfig.betting_config_name == config_name
        ).first()

        if not betting_config:
            betting_config = BettingConfig(betting_config_name=config_name)
            db.add(betting_config)
            db.flush()

            sb_detail = BettingConfigDetails(
                betting_config_id=betting_config.betting_config_id,
                bet_name="SB",
                bet_amount=1,
            )
            
            bb_detail = BettingConfigDetails(
                betting_config_id=betting_config.betting_config_id,
                bet_name="BB",
                bet_amount=2,
            )
            db.add_all([sb_detail, bb_detail])
            print(f"✓ Created Betting Config '{config_name}' with SB=1, BB=2")
        else:
            print(f"✓ Betting Config '{config_name}' already exists")
            
        db.commit()

    except Exception as e:
        print(f"Error during seeding: {e}")
        db.rollback()

    finally:
        db.close()





if __name__ == "__main__":
    Base.metadata.create_all(bind=engine)
    seed_player_tables()
    seed_betting_config_tables()
