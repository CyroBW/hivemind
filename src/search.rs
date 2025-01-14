mod alpha_beta;
mod defs;
pub mod eval;
mod history;
mod iter_deep;
mod qsearch;
mod sorting;

use crate::board::Board;
use crate::transposition::TranspositionTable;
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

use crossbeam_channel::unbounded;
use crossbeam_channel::Sender;

use crate::types::parameters::Parameters;
use defs::{SearchInfo, SearchParams, SearchRefs};
use shakmaty::zobrist::Zobrist64;
use shakmaty::Move;

pub struct Search {
    handle: Option<JoinHandle<()>>,
    sender: Option<Sender<String>>,
}

impl Search {
    pub fn new() -> Self {
        Self {
            handle: None,
            sender: None,
        }
    }

    pub fn init(&mut self, mtx_board: Arc<Mutex<Board>>, mtx_tt: Arc<Mutex<TranspositionTable>>) {
        let (s, r) = unbounded::<String>();
        let h = thread::spawn(move || {
            let mut search_params = SearchParams {
                depth: 69,
                search_time: 600000,
            };
            let mut quit = false;
            let mut halt = true;

            while !quit {
                let cmd = r.recv().unwrap();
                let board = mtx_board.lock().unwrap();
                let mut tt = mtx_tt.lock().unwrap();

                if cmd.starts_with("movetime") {
                    if let Some(time_left) = cmd.split_whitespace().nth(1) {
                        if let Ok(time_left) = time_left.parse::<u128>() {
                            search_params.search_time = time_left;
                        }
                    }
                }

                match cmd.as_str() {
                    "go" => halt = false,
                    "stop" => halt = true,
                    "quit" => quit = true,
                    _ => (),
                }

                if !halt && !quit {
                    let mut search_info = SearchInfo::new();

                    let mut search_refs = SearchRefs {
                        board: &mut board.clone(),
                        params: Parameters::default(),
                        search_params: &mut search_params,
                        search_info: &mut search_info,
                        tt: &mut tt,
                        tt_enabled: true,
                    };

                    let best_move = Search::iterative_deepening(&mut search_refs);
                    if let Some(mv) = best_move {
                        println!("bestmove {}", search_refs.board.to_uci(&mv));
                    } else {
                        println!("bestmove (none)");
                    }

                    halt = true;
                }
            }
        });

        self.handle = Some(h);
        self.sender = Some(s);
    }

    pub fn send(&mut self, cmd: String) {
        if let Some(s) = &self.sender {
            s.send(cmd).expect("Broken channel");
        }
    }
}
