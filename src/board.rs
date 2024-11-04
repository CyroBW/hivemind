use crate::nnue::Network;
use crate::types::Score;
use shakmaty::{
    fen::{Fen, ParseFenError},
    uci::UciMove,
    zobrist::{Zobrist64, ZobristHash},
    CastlingMode, Color, EnPassantMode, Move, MoveList, Position, Role, Square,
};

use shakmaty::variant::Crazyhouse;
pub const PIECE_VALUES: [i32; 7] = [
    0,   // 0
    180, // Pawn
    478, // Knight
    478, // Bishop
    642, // Rook
    900, // Queen
    0,   // King
];

#[derive(Clone, Default)]
pub struct Board {
    pos: Crazyhouse,
    nnue: Network,
    state_stack: Vec<Crazyhouse>,
    move_stack: Vec<Option<Move>>,
    history: Vec<u64>,
}

impl Board {
    pub fn new(fen: &str) -> Result<Self, ParseFenError> {
        let fen_string = String::from(fen);
        let fen: Fen = fen_string.parse()?;
        let pos: Crazyhouse = fen.into_position(CastlingMode::Standard).unwrap();
        let mut nnue = Network::default();
        for color in [Color::White, Color::Black] {
            for piece in Role::ALL {
                for square in pos.board().by_color(color) & pos.board().by_role(piece) {
                    nnue.accumulate(color, piece, square);
                }
            }
        }
        let state_stack = Vec::default();
        let move_stack = Vec::default();
        let history = Vec::default();
        Ok(Self {
            pos,
            nnue,
            state_stack,
            move_stack,
            history,
        })
    }
    pub fn starting_position() -> Self {
        Self::new("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap()
    }
    pub fn parse_uci(&self, uci_move: &str) -> Move {
        let uci: UciMove = uci_move.parse().unwrap();
        uci.to_move(&self.pos).unwrap()
    }

    pub fn play_uci(&mut self, uci_move: &str) {
        let moves = self.legal_moves();
        for mv in moves {
            if self.to_uci(&mv).to_string() == uci_move {
                self.make_move::<true>(&mv);
                break;
            }
        }
    }
    pub fn get_hash(&self) -> u64 {
        self.pos.zobrist_hash::<Zobrist64>(EnPassantMode::Legal).0
    }

    pub fn to_uci(&self, mv: &Move) -> UciMove {
        mv.to_uci(self.pos.castles().mode())
    }

    pub fn turn(&self) -> Color {
        self.pos.turn()
    }

    pub fn in_check(&self) -> bool {
        self.pos.is_check()
    }

    pub fn capture_moves(&self) -> MoveList {
        self.pos.capture_moves()
    }

    pub fn state(&self) -> Crazyhouse {
        self.pos.clone()
    }

    pub fn make_move<const IN_PLACE: bool>(&mut self, mv: &Move) {
        self.state_stack.push(self.pos.clone());
        if !IN_PLACE {
            self.nnue.push();
        }

        let stm = self.pos.turn();
        match mv {
            Move::Normal {
                role,
                from,
                capture,
                to,
                promotion,
            } => {
                self.remove_piece(stm, *role, *from);

                if let Some(capture) = mv.capture() {
                    self.remove_piece(stm.other(), capture, *to);
                }

                if let Some(promotion) = mv.promotion() {
                    self.add_piece(stm, promotion, *to);
                } else {
                    self.add_piece(stm, *role, *to);
                }
            }
            Move::EnPassant { from, to } => {
                self.remove_piece(stm, Role::Pawn, *from);
                self.add_piece(stm, Role::Pawn, *to);

                let target = to.xor(Square::A2);
                if let Some(capture) = self.pos.board().role_at(target) {
                    self.remove_piece(stm.other(), capture, target);
                }
            }
            Move::Castle { king, rook } => {
                self.remove_piece(stm, Role::King, *king);
                self.remove_piece(stm, Role::Rook, *rook);
                match king {
                    Square::E1 => {
                        // White kingside castling
                        if rook == &Square::H1 {
                            self.add_piece(stm, Role::King, Square::G1);
                            self.add_piece(stm, Role::Rook, Square::F1);
                        }
                        // White queenside castling
                        else if rook == &Square::A1 {
                            self.add_piece(stm, Role::King, Square::C1);
                            self.add_piece(stm, Role::Rook, Square::D1);
                        }
                    }
                    Square::E8 => {
                        // Black kingside castling
                        if rook == &Square::H8 {
                            self.add_piece(stm, Role::King, Square::G8);
                            self.add_piece(stm, Role::Rook, Square::F8);
                        }
                        // Black queenside castling
                        else if rook == &Square::A8 {
                            self.add_piece(stm, Role::King, Square::C8);
                            self.add_piece(stm, Role::Rook, Square::D8);
                        }
                    }
                    _ => (),
                }
            }
            Move::Put { role, to } => {
                self.add_piece(stm, *role, *to);
            }
        }

        self.pos.play_unchecked(mv);
        self.nnue.commit();

        self.move_stack.push(Some(mv.clone()));
        self.history.push(self.get_hash());
    }
    pub fn undo_move(&mut self) {
        self.nnue.pop();
        let _mv = self.move_stack.pop();
        self.pos = self.state_stack.pop().unwrap();
        self.history.pop();
    }

    pub fn make_null_move(&mut self) {
        self.state_stack.push(self.pos.clone());
        if let Ok(pos) = self.pos.clone().swap_turn() {
            self.pos = pos;
        }
        self.move_stack.push(None);
    }

    pub fn undo_null_move(&mut self) {
        self.pos = self.state_stack.pop().unwrap();
        let _mv = self.move_stack.pop();
    }

    pub fn is_last_move_null(&self) -> bool {
        self.move_stack.last().is_none()
    }

    pub fn evaluate(&self) -> i32 {
        let stm = self.pos.turn();
        let mut eval = self.nnue.evaluate(stm);
        if let Some(pockets) = self.pos.pockets() {
            for role in Role::ALL {
                eval += *pockets.get(stm).get(role) as i32 * PIECE_VALUES[role as usize];
                eval -= *pockets.get(stm.other()).get(role) as i32 * PIECE_VALUES[role as usize];
            }
        }
        eval.clamp(-Score::MATE_BOUND + 1, Score::MATE_BOUND - 1)
    }

    pub fn legal_moves(&self) -> MoveList {
        self.pos.legal_moves()
    }

    pub fn three_fold(&self) -> bool {
        let mut cnt = 0;
        let hash = self.get_hash();
        for &h in self.history.iter().rev() {
            if h == hash {
                cnt += 1;
                if cnt > 1 {
                    return true;
                }
            }
        }
        false
    }

    fn add_piece(&mut self, color: Color, piece: Role, square: Square) {
        self.nnue.activate(color, piece, square);
    }
    fn remove_piece(&mut self, color: Color, piece: Role, square: Square) {
        self.nnue.deactivate(color, piece, square);
    }
}

#[cfg(test)]
mod tests {
    use crate::board::Board;
    #[test]
    fn test_nnue() {
        let board = Board::starting_position();
        assert_eq!(board.evaluate(), 48);
    }
}
