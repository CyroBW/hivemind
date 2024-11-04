use super::sorting::see;
use crate::transposition::Bound;
use crate::types::{parameters::*, Score, MAX_PLY};

use super::{defs::SearchRefs, Search};
use shakmaty::{Move, MoveList};

impl Search {
    pub fn alpha_beta(refs: &mut SearchRefs, mut depth: i32, mut alpha: i32, mut beta: i32) -> i32 {
        let ply = refs.search_info.ply as usize;
        refs.search_info.pv[ply].fill(None);

        if (refs.search_info.nodes & 2047) == 0
            && refs.search_info.elapsed() > refs.search_params.search_time
        {
            refs.search_info.terminated = true;
            return 0;
        }

        let is_root = ply == 0;
        let pv_node = beta - alpha > 1;
        let original_alpha = alpha;
        let in_check = refs.board.in_check();

        let mut best_score: i32 = -Score::INFINITY;
        let mut best_move: Option<Move> = None;

        let mut captures = MoveList::default();
        let mut quiets = MoveList::default();

        if !is_root {
            // Draw Detection
            if refs.board.three_fold() {
                return Score::DRAW;
            }

            // Mate Distance Pruning
            alpha = alpha.max(-Score::MATE + ply as i32);
            beta = beta.min(Score::MATE - (ply as i32) - 1);
            if alpha >= beta {
                return alpha;
            }
        }

        if ply >= MAX_PLY - 1 {
            return refs.board.evaluate();
        }
        if depth <= 0 && !in_check {
            return Search::qsearch(refs, alpha, beta);
        }
        depth = depth.max(0);

        let mut tt_move: Option<Move> = None;
        let hit = refs.tt.read(refs.board.get_hash(), ply);
        if let Some(hit) = hit {
            if !pv_node && hit.valid_cutoff(alpha, beta, depth) {
                return hit.score;
            }
            tt_move = hit.m;
        }

        // Internal Iterative Reductions
        if !is_root && tt_move.is_none() && depth >= iir_depth() {
            depth -= 1;
        }

        if in_check {
            depth += 1;
        }

        refs.search_info.nodes += 1;
        refs.search_info.sel_depth = refs.search_info.sel_depth.max(ply);
        refs.search_info.pv_length[ply] = ply;

        let eval = refs.board.evaluate();
        if !in_check && !pv_node && !is_root {
            // Reverse Futility Pruning
            if depth < rfp_depth() && eval - rfp_margin() * depth > beta {
                return eval;
            }
            // Razoring
            if depth <= razoring_depth()
                && eval + razoring_margin() * depth + razoring_fixed_margin() <= alpha
            {
                let score = Search::qsearch(refs, alpha, beta);
                if score <= alpha {
                    return score;
                }
            }
            // Null move pruning
            if !refs.board.is_last_move_null() && depth >= 4 && eval > beta {
                let r = 3 + depth / 3 + ((eval - beta) / 200).min(4);

                refs.board.make_null_move();
                let score = -Search::alpha_beta(refs, depth - r, -beta, -beta + 1);
                refs.board.undo_null_move();

                if score >= beta {
                    return beta;
                }
            }
        }

        let mut moves = refs.board.legal_moves();
        Search::sort_moves(&mut moves, &refs.search_info.pv[ply][ply], &tt_move, refs);

        for (moves_searched, mv) in (&moves).into_iter().enumerate() {
            if !is_root && moves_searched > 0 && alpha > -Score::MATE_BOUND {
                // Futility Pruning
                if !pv_node
                    && !in_check
                    && !mv.is_capture()
                    && depth <= fp_depth()
                    && eval + fp_margin() * depth + fp_fixed_margin() < alpha
                {
                    break;
                }

                // Static Exchange Evaluation Pruning. Skip moves that are losing material.
                if depth < see_depth()
                    && !see(
                        &refs.board.state(),
                        mv,
                        -[see_quiet_margin(), see_noisy_margin()][mv.is_capture() as usize] * depth,
                    )
                    .expect("Error evaluationg SEE")
                {
                    continue;
                }
            }

            refs.board.make_move::<false>(mv);
            refs.search_info.ply += 1;
            refs.tt.prefetch(refs.board.get_hash());

            let mut score;
            if moves_searched == 0 {
                // We always search full depth on hypothesis best move
                score = -Search::alpha_beta(refs, depth - 1, -beta, -alpha);
            } else {
                // Late Move Reductions - try to fail low
                if moves_searched >= LMR_MOVES_PLAYED
                    && depth >= LMR_DEPTH
                    && ply >= 3
                    && !mv.is_capture()
                    && !mv.is_promotion()
                    && !in_check
                    && Some(mv) != refs.search_info.killers[ply].as_ref()
                {
                    score = -Search::alpha_beta(refs, depth - 2, -alpha - 1, -alpha);
                } else {
                    // When we don't do LMR we don't fail low
                    score = alpha + 1;
                }
                if score > alpha {
                    score = -Search::alpha_beta(refs, depth - 1, -alpha - 1, -alpha);
                    if score > alpha && score < beta {
                        // We found a better move so re-search
                        score = -Search::alpha_beta(refs, depth - 1, -beta, -alpha);
                    }
                }
            }

            refs.board.undo_move();
            refs.search_info.ply -= 1;

            if score > best_score {
                best_score = score;
                best_move = Some(mv.clone());

                if score > alpha {
                    alpha = score;
                    Search::update_pv(refs, best_move.clone(), ply);
                }
            }

            if alpha >= beta {
                break;
            }

            if mv.is_capture() {
                captures.push(mv.clone());
            } else {
                quiets.push(mv.clone());
            }
        }

        if moves.is_empty() {
            return if in_check {
                Score::mated_in(ply)
            } else {
                Score::DRAW
            };
        }

        let bound = match best_score {
            s if s <= original_alpha => Bound::Alpha,
            s if s >= beta => Bound::Beta,
            _ => Bound::Exact,
        };
        if bound == Bound::Beta {
            Search::update_ordering_heuristics(
                refs,
                depth,
                best_move.clone().expect("Move should not be None"),
                captures,
                quiets,
            );
        }

        refs.tt.write(
            refs.board.get_hash(),
            depth,
            best_score,
            bound,
            best_move,
            ply,
        );
        best_score
    }

    pub fn update_ordering_heuristics(
        refs: &mut SearchRefs,
        depth: i32,
        best_move: Move,
        captures: MoveList,
        quiets: MoveList,
    ) {
        if best_move.is_capture() {
            refs.search_info
                .history
                .update_capture(refs.board.state(), best_move, captures, depth);
        } else {
            refs.search_info.killers[refs.search_info.ply as usize] = Some(best_move.clone());
            refs.search_info
                .history
                .update_main(refs.board.turn(), best_move, quiets, depth);
        }
    }

    pub fn update_pv(refs: &mut SearchRefs, best_move: Option<Move>, ply: usize) {
        refs.search_info.pv[ply][ply] = best_move.clone();
        for next_ply in ply + 1..refs.search_info.pv_length[ply + 1] {
            refs.search_info.pv[ply][next_ply] = refs.search_info.pv[ply + 1][next_ply].clone();
        }
        refs.search_info.pv_length[ply] = refs.search_info.pv_length[ply + 1];
    }
}
