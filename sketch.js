/**
 * Neuroevolution Tic-Tac-Toe (p5.js)
 *
 * Tunables:
 * POP, HIDDEN_SIZE, GAMES_PER_BRAIN,
 * MUTATION_RATE, MUTATION_STRENGTH, ELITE_COUNT,
 * WIN_SCORE, TIE_SCORE, TRAINING_SPEED_* plies, AUTO_GEN_INTERVAL_MS
 */
/** Canvas edge length (square). Layout scales with S = CANVAS / 600. */
const CANVAS = 900;
const SPLIT_X = CANVAS * 0.5;
const S = CANVAS / 600;

/** Right-panel HUD (scaled). Extra gap before NN column titles. */
const HUD_X = SPLIT_X + 18 * S;
const HUD_Y_GEN = 18 * S;
const HUD_Y_FIT = 42 * S;
const HUD_Y_ROW3 = 62 * S;
const HUD_Y_ROW4 = 78 * S;
const HUD_Y_R = 94 * S;
const HUD_Y_SPEED = 110 * S;
const HUD_TEXT_BOTTOM = HUD_Y_SPEED + 13 * S;
const VIZ_CAPTION_BASELINE = HUD_TEXT_BOTTOM + 44 * S;
/** Space from column-title baseline down to first row of nodes (keep small so titles sit near the net). */
const VIZ_TOP = VIZ_CAPTION_BASELINE + 7 * S;
const VIZ_BOT = CANVAS - 55 * S;
/** Left-pane hint text wraps within this width so it does not cross SPLIT_X. */
const LEFT_PANEL_TEXT_W = SPLIT_X - 22 * S;

const POP = 100;
const HIDDEN_SIZE = 14;
const GAMES_PER_BRAIN = 15;
const MUTATION_RATE = 0.12;
const MUTATION_STRENGTH = 0.35;
const ELITE_COUNT = 2;
const WIN_SCORE = 1;
const TIE_SCORE = 0.5;
/** Plies simulated per draw() frame at each speed tier. -1 means "Turbo" (time-budgeted). */
const TRAINING_SPEED_PLIES = [1, 12, 160, -1];
const TRAINING_SPEED_LABELS = ["slow", "medium", "fast", "turbo"];
const AUTO_GEN_INTERVAL_MS = 1200;

const LINE_WEIGHT = 2 * S;
const PAD_FR = 0.18;

// --- Board / game ---------------------------------------------------------

class Board {
  constructor() {
    this.cells = [
      [null, null, null],
      [null, null, null],
      [null, null, null],
    ];
  }

  isFull() {
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 3; c++) {
        if (this.cells[r][c] === null) return false;
      }
    }
    return true;
  }

  checkWin(player) {
    for (let r = 0; r < 3; r++) {
      if (
        this.cells[r][0] === player &&
        this.cells[r][1] === player &&
        this.cells[r][2] === player
      )
        return true;
    }
    for (let c = 0; c < 3; c++) {
      if (
        this.cells[0][c] === player &&
        this.cells[1][c] === player &&
        this.cells[2][c] === player
      )
        return true;
    }
    if (
      this.cells[0][0] === player &&
      this.cells[1][1] === player &&
      this.cells[2][2] === player
    )
      return true;
    if (
      this.cells[0][2] === player &&
      this.cells[1][1] === player &&
      this.cells[2][0] === player
    )
      return true;
    return false;
  }

  toInput(myPiece) {
    const input = [];
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 3; c++) {
        const cell = this.cells[r][c];
        if (cell === null) input.push(0);
        else if (cell === myPiece) input.push(1);
        else input.push(-1);
      }
    }
    return input;
  }

  getEmptyIndices() {
    const empties = [];
    for (let i = 0; i < 9; i++) {
      const r = Math.floor(i / 3);
      const c = i % 3;
      if (this.cells[r][c] === null) empties.push(i);
    }
    return empties;
  }

  makeMove(index, player) {
    const r = Math.floor(index / 3);
    const c = index % 3;
    if (this.cells[r][c] === null) {
      this.cells[r][c] = player;
      return true;
    }
    return false;
  }
}

function randomLegalIndex(board) {
  const empties = board.getEmptyIndices();
  if (empties.length === 0) return -1;
  return empties[Math.floor(random(empties.length))];
}

function pickNetMove(net, board, myPiece) {
  const input = board.toInput(myPiece);
  const { out } = net.forward(input);
  let best = -1;
  let bestScore = -Infinity;
  const empties = board.getEmptyIndices();

  for (const i of empties) {
    if (out[i] > bestScore) {
      bestScore = out[i];
      best = i;
    }
  }
  return best;
}

// --- Neural network ---------------------------------------------------------

class NeuralNet {
  constructor() {
    this.hiddenSize = HIDDEN_SIZE;
    this.wIH = [];
    this.bH = [];
    for (let h = 0; h < this.hiddenSize; h++) {
      this.wIH[h] = [];
      for (let i = 0; i < 9; i++) {
        this.wIH[h][i] = random(-0.5, 0.5);
      }
      this.bH[h] = random(-0.5, 0.5);
    }
    this.wHO = [];
    this.bO = [];
    for (let o = 0; o < 9; o++) {
      this.wHO[o] = [];
      for (let h = 0; h < this.hiddenSize; h++) {
        this.wHO[o][h] = random(-0.5, 0.5);
      }
      this.bO[o] = random(-0.5, 0.5);
    }
  }

  forward(input) {
    const hidden = [];
    for (let h = 0; h < this.hiddenSize; h++) {
      let sum = this.bH[h];
      for (let i = 0; i < 9; i++) {
        sum += this.wIH[h][i] * input[i];
      }
      hidden[h] = Math.tanh(sum);
    }
    const out = [];
    for (let o = 0; o < 9; o++) {
      let sum = this.bO[o];
      for (let h = 0; h < this.hiddenSize; h++) {
        sum += this.wHO[o][h] * hidden[h];
      }
      out[o] = sum;
    }
    return { hidden, out };
  }

  copy() {
    const n = new NeuralNet();
    for (let h = 0; h < this.hiddenSize; h++) {
      for (let i = 0; i < 9; i++) {
        n.wIH[h][i] = this.wIH[h][i];
      }
      n.bH[h] = this.bH[h];
    }
    for (let o = 0; o < 9; o++) {
      for (let h = 0; h < this.hiddenSize; h++) {
        n.wHO[o][h] = this.wHO[o][h];
      }
      n.bO[o] = this.bO[o];
    }
    return n;
  }

  mutate(rate, strength) {
    for (let h = 0; h < this.hiddenSize; h++) {
      for (let i = 0; i < 9; i++) {
        if (random() < rate) {
          this.wIH[h][i] += randomGaussian() * strength;
        }
      }
      if (random() < rate) {
        this.bH[h] += randomGaussian() * strength;
      }
    }
    for (let o = 0; o < 9; o++) {
      for (let h = 0; h < this.hiddenSize; h++) {
        if (random() < rate) {
          this.wHO[o][h] += randomGaussian() * strength;
        }
      }
      if (random() < rate) {
        this.bO[o] += randomGaussian() * strength;
      }
    }
  }

  serialize() {
    return JSON.stringify({
      hiddenSize: this.hiddenSize,
      wIH: this.wIH,
      bH: this.bH,
      wHO: this.wHO,
      bO: this.bO,
    });
  }

  static deserialize(data) {
    if (typeof data === "string") data = JSON.parse(data);
    const n = new NeuralNet();
    n.hiddenSize = data.hiddenSize;
    n.wIH = data.wIH;
    n.bH = data.bH;
    n.wHO = data.wHO;
    n.bO = data.bO;
    return n;
  }
}

// --- Genetic algorithm & Training -------------------------------------------

class EvolutionaryTrainer {
  constructor() {
    this.population = [];
    this.generation = 0;
    this.bestFitnessThisGen = 0;
    this.showcaseBrain = null;

    this.trainingActive = false;
    this.fitnessesWorking = [];
    this.trainInd = 0;
    this.trainGame = 0;
    this.visWins = 0;
    this.visTies = 0;
    this.visBoard = null;
    this.visTurn = "X";
    this.visNetIsX = true;

    this.initPopulation();
  }

  initPopulation() {
    this.population = [];
    for (let i = 0; i < POP; i++) {
      this.population.push(new NeuralNet());
    }
    this.generation = 0;
    this.bestFitnessThisGen = 0;
    this.trainingActive = false;
    this.visBoard = new Board();

    if (this.loadFromLocalStorage()) {
      console.log("Loaded existing brain from localStorage");
    } else if (this.population.length > 0) {
      this.showcaseBrain = this.population[0].copy();
    }
  }

  saveToLocalStorage() {
    if (!this.showcaseBrain) return;
    const data = {
      generation: this.generation,
      bestFitness: this.bestFitnessThisGen,
      brain: this.showcaseBrain.serialize(),
    };
    localStorage.setItem("tictactoe_evolution_data", JSON.stringify(data));
  }

  loadFromLocalStorage() {
    const raw = localStorage.getItem("tictactoe_evolution_data");
    if (!raw) return false;
    try {
      const data = JSON.parse(raw);
      this.generation = data.generation || 0;
      this.bestFitnessThisGen = data.bestFitness || 0;
      this.showcaseBrain = NeuralNet.deserialize(data.brain);
      // Fill population with mutants of the loaded champion to continue evolution
      this.population = [];
      for (let i = 0; i < POP; i++) {
        const child = this.showcaseBrain.copy();
        if (i > 0) child.mutate(MUTATION_RATE, MUTATION_STRENGTH);
        this.population.push(child);
      }
      return true;
    } catch (e) {
      console.error("Failed to load from localStorage", e);
      return false;
    }
  }

  startGenerationTraining() {
    if (this.trainingActive) return;
    this.trainingActive = true;
    this.trainInd = 0;
    this.trainGame = 0;
    this.visWins = 0;
    this.visTies = 0;
    this.fitnessesWorking = new Array(POP).fill(0);
    this.resetVisualGame();
  }

  resetVisualGame() {
    this.visBoard = new Board();
    this.visNetIsX = random() < 0.5;
    this.visTurn = "X";
  }

  advanceVisualTrainingPly() {
    if (!this.trainingActive || !this.visBoard) return;

    const net = this.population[this.trainInd];
    const isNet =
      (this.visTurn === "X" && this.visNetIsX) ||
      (this.visTurn === "O" && !this.visNetIsX);

    let idx;
    if (isNet) {
      idx = pickNetMove(net, this.visBoard, this.visTurn);
      if (idx < 0) idx = randomLegalIndex(this.visBoard);
    } else {
      idx = randomLegalIndex(this.visBoard);
    }

    if (idx !== -1) {
      this.visBoard.makeMove(idx, this.visTurn);
    }

    if (this.visBoard.checkWin(this.visTurn)) {
      if (isNet) this.visWins++;
      this.finishCurrentTrainingGame();
      return;
    }
    if (this.visBoard.isFull()) {
      this.visTies++;
      this.finishCurrentTrainingGame();
      return;
    }
    this.visTurn = this.visTurn === "X" ? "O" : "X";
  }

  finishCurrentTrainingGame() {
    this.trainGame++;
    if (this.trainGame >= GAMES_PER_BRAIN) {
      this.fitnessesWorking[this.trainInd] =
        this.visWins * WIN_SCORE + this.visTies * TIE_SCORE;
      this.trainInd++;
      this.trainGame = 0;
      this.visWins = 0;
      this.visTies = 0;
      if (this.trainInd >= POP) {
        this.completeBreeding(this.fitnessesWorking);
        this.trainingActive = false;
        this.visBoard = new Board();
        this.visTurn = "X";
        return;
      }
    }
    this.resetVisualGame();
  }

  completeBreeding(fitnesses) {
    const idx = Array.from({ length: POP }, (_, i) => i);
    idx.sort((a, b) => fitnesses[b] - fitnesses[a]);

    const sorted = idx.map((i) => this.population[i]);

    this.bestFitnessThisGen = fitnesses[idx[0]];
    this.showcaseBrain = sorted[0].copy();

    const next = [];
    for (let e = 0; e < ELITE_COUNT; e++) {
      next.push(sorted[e].copy());
    }
    const topHalf = Math.max(1, Math.ceil(POP / 2));
    while (next.length < POP) {
      const parent = sorted[Math.floor(random(topHalf))];
      const child = parent.copy();
      child.mutate(MUTATION_RATE, MUTATION_STRENGTH);
      next.push(child);
    }
    this.population = next;
    this.generation++;
    this.saveToLocalStorage();
  }

  clearStorage() {
    localStorage.removeItem("tictactoe_evolution_data");
  }
}

const AppState = {
  TRAINING: "TRAINING",
  PLAYING: "PLAYING",
  IDLE: "IDLE",
};

class GameManager {
  constructor() {
    this.trainer = new EvolutionaryTrainer();
    this.state = AppState.IDLE;
    this.autoTrain = false;
    this.lastAutoGenMs = millis();
    this.trainingSpeedIndex = 0;

    // Batch training state
    this.batchTargetGen = 0;

    // Player vs AI state
    this.humanBoard = null;
    this.humanPiece = "X";
    this.aiPiece = "O";
    this.humanTurn = true;
    this.gameResult = null; // "X", "O", or "Tie"
  }

  update() {
    // If we have a batch target, keep starting training until we hit it
    if (this.batchTargetGen > this.trainer.generation) {
      if (this.state !== AppState.TRAINING) {
        this.startTraining();
      }
    }

    if (
      this.autoTrain &&
      this.state !== AppState.TRAINING &&
      this.state !== AppState.PLAYING &&
      millis() - this.lastAutoGenMs >= AUTO_GEN_INTERVAL_MS
    ) {
      this.startTraining();
    }

    if (this.state === AppState.TRAINING) {
      const pLimit = TRAINING_SPEED_PLIES[this.trainingSpeedIndex];
      const startTime = performance.now();
      let pCount = 0;

      while (true) {
        const wasActive = this.trainer.trainingActive;
        this.trainer.advanceVisualTrainingPly();

        if (!this.trainer.trainingActive && wasActive) {
          this.state = AppState.IDLE;
          this.lastAutoGenMs = millis();
          break;
        }

        pCount++;
        // If fixed limit mode (slow/med/fast)
        if (pLimit !== -1 && pCount >= pLimit) break;
        // If Turbo mode, limit by time (e.g. 25ms budget per frame)
        if (pLimit === -1 && performance.now() - startTime > 25) break;
      }
    }

    if (this.state === AppState.PLAYING && !this.humanTurn && !this.gameResult) {
      // Small delay so AI doesn't move instantly
      if (frameCount % 30 === 0) {
        this.aiMove();
      }
    }
  }

  startHumanGame() {
    this.state = AppState.PLAYING;
    this.humanBoard = new Board();
    this.humanPiece = random() < 0.5 ? "X" : "O";
    this.aiPiece = this.humanPiece === "X" ? "O" : "X";
    this.humanTurn = this.humanPiece === "X";
    this.gameResult = null;
  }

  handleMouseClick(mx, my, ox, oy, boardPx) {
    if (this.state !== AppState.PLAYING || !this.humanTurn || this.gameResult)
      return;

    const cell = boardPx / 3;
    const c = Math.floor((mx - ox) / cell);
    const r = Math.floor((my - oy) / cell);

    if (r >= 0 && r < 3 && c >= 0 && c < 3) {
      const idx = r * 3 + c;
      if (this.humanBoard.makeMove(idx, this.humanPiece)) {
        this.checkGameEnd();
        this.humanTurn = false;
      }
    }
  }

  aiMove() {
    if (!this.trainer.showcaseBrain) return;
    const idx = pickNetMove(
      this.trainer.showcaseBrain,
      this.humanBoard,
      this.aiPiece
    );
    if (idx !== -1) {
      this.humanBoard.makeMove(idx, this.aiPiece);
    } else {
      // Fallback to random if AI is confused
      const ridx = randomLegalIndex(this.humanBoard);
      if (ridx !== -1) this.humanBoard.makeMove(ridx, this.aiPiece);
    }
    this.checkGameEnd();
    this.humanTurn = true;
  }

  checkGameEnd() {
    if (this.humanBoard.checkWin(this.humanPiece)) this.gameResult = "You Win!";
    else if (this.humanBoard.checkWin(this.aiPiece)) this.gameResult = "AI Wins!";
    else if (this.humanBoard.isFull()) this.gameResult = "Tie!";
  }

  startTraining() {
    // Force switch to TRAINING state
    this.state = AppState.TRAINING;
    this.humanBoard = null;
    this.gameResult = null;
    this.trainer.startGenerationTraining();
  }

  startBatch(count) {
    this.batchTargetGen = this.trainer.generation + count;
    this.setSpeed(2); // Set to fast for batches
    this.startTraining();
  }

  toggleAuto() {
    this.autoTrain = !this.autoTrain;
    this.lastAutoGenMs = millis();
    if (!this.autoTrain) {
      this.batchTargetGen = 0;
    }
  }

  reset() {
    this.trainer.clearStorage();
    this.trainer.initPopulation();
    this.state = AppState.IDLE;
    this.autoTrain = false;
    this.batchTargetGen = 0;
    this.lastAutoGenMs = millis();
    this.humanBoard = null;
    this.gameResult = null;
  }

  setSpeed(index) {
    this.trainingSpeedIndex = constrain(
      index,
      0,
      TRAINING_SPEED_LABELS.length - 1
    );
  }

  cycleSpeed(dir) {
    this.trainingSpeedIndex =
      (this.trainingSpeedIndex + dir + TRAINING_SPEED_LABELS.length) %
      TRAINING_SPEED_LABELS.length;
  }
}

let mgr;

let inputNodePos = [];
let hiddenNodePos = [];
let outputNodePos = [];

function layoutNetNodes() {
  inputNodePos = [];
  hiddenNodePos = [];
  outputNodePos = [];
  const xIn = SPLIT_X + 45 * S;
  const xH = SPLIT_X + SPLIT_X / 2;
  const xOut = CANVAS - 45 * S;
  for (let i = 0; i < 9; i++) {
    const y = VIZ_TOP + ((i + 1) * (VIZ_BOT - VIZ_TOP)) / 10;
    inputNodePos.push({ x: xIn, y: y });
  }
  for (let h = 0; h < HIDDEN_SIZE; h++) {
    const y =
      VIZ_TOP + ((h + 1) * (VIZ_BOT - VIZ_TOP)) / (HIDDEN_SIZE + 1);
    hiddenNodePos.push({ x: xH, y: y });
  }
  for (let o = 0; o < 9; o++) {
    const y = VIZ_TOP + ((o + 1) * (VIZ_BOT - VIZ_TOP)) / 10;
    outputNodePos.push({ x: xOut, y: y });
  }
}

function drawNetColumnCaptions() {
  noStroke();
  textAlign(CENTER, BOTTOM);
  textSize(11 * S);
  fill(150, 152, 165);
  const yCap = VIZ_CAPTION_BASELINE;
  text("Board cells (9)", inputNodePos[4].x, yCap);
  text("Hidden (" + HIDDEN_SIZE + ")", hiddenNodePos[7].x, yCap);
  text("Move scores (9)", outputNodePos[4].x, yCap);
}

function drawNetNodeKey() {
  textAlign(LEFT, TOP);
  textSize(10 * S);
  fill(110, 112, 125);
  text(
    "Top→bottom: rows 1–3 of board, 3 cells each.",
    SPLIT_X + 12 * S,
    VIZ_BOT + 8 * S
  );
  text(
    "Outputs = score to play there (only empty squares matter).",
    SPLIT_X + 12 * S,
    VIZ_BOT + 22 * S
  );
}

function activationColor(t) {
  return constrain(128 + t * 80, 40, 220);
}

function drawNetworkViz(net, input, hidden, out, boardForLegal) {
  for (let h = 0; h < HIDDEN_SIZE; h++) {
    for (let i = 0; i < 9; i++) {
      const w = net.wIH[h][i];
      const a = constrain(abs(w) * 80, 20, 100);
      if (w >= 0) stroke(80, 120, 90, a);
      else stroke(120, 80, 90, a);
      strokeWeight(0.5 * S);
      line(
        inputNodePos[i].x,
        inputNodePos[i].y,
        hiddenNodePos[h].x,
        hiddenNodePos[h].y
      );
    }
  }
  for (let o = 0; o < 9; o++) {
    for (let h = 0; h < HIDDEN_SIZE; h++) {
      const w = net.wHO[o][h];
      const a = constrain(abs(w) * 80, 20, 100);
      if (w >= 0) stroke(90, 100, 130, a);
      else stroke(130, 90, 100, a);
      strokeWeight(0.5 * S);
      line(
        hiddenNodePos[h].x,
        hiddenNodePos[h].y,
        outputNodePos[o].x,
        outputNodePos[o].y
      );
    }
  }
  const dIn = 10 * S;
  const dHid = 9 * S;
  const dOut = 10 * S;
  const dRing = 14 * S;
  noStroke();
  for (let i = 0; i < 9; i++) {
    fill(activationColor(input[i]));
    ellipse(inputNodePos[i].x, inputNodePos[i].y, dIn, dIn);
  }
  for (let h = 0; h < HIDDEN_SIZE; h++) {
    fill(activationColor(hidden[h]));
    ellipse(hiddenNodePos[h].x, hiddenNodePos[h].y, dHid, dHid);
  }
  for (let o = 0; o < 9; o++) {
    fill(activationColor(out[o] * 0.15));
    ellipse(outputNodePos[o].x, outputNodePos[o].y, dOut, dOut);
  }

  if (boardForLegal !== null) {
    for (let o = 0; o < 9; o++) {
      const r = Math.floor(o / 3);
      const c = o % 3;
      if (boardForLegal.cells[r][c] === null) {
        noFill();
        stroke(90, 160, 110, 140);
        strokeWeight(1 * S);
        ellipse(outputNodePos[o].x, outputNodePos[o].y, dRing, dRing);
      }
    }
  }
}

function drawBoardRegion(board, ox, oy, boardPx) {
  const cell = boardPx / 3;
  const pad = cell * PAD_FR;
  stroke(90, 92, 102);
  strokeWeight(LINE_WEIGHT);
  noFill();
  for (let i = 1; i < 3; i++) {
    const gx = ox + i * cell;
    line(gx, oy, gx, oy + boardPx);
    const gy = oy + i * cell;
    line(ox, gy, ox + boardPx, gy);
  }
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      const mark = board.cells[r][c];
      const x = ox + c * cell;
      const y = oy + r * cell;
      if (mark === "X") {
        stroke(220, 225, 235);
        strokeWeight(LINE_WEIGHT + 1);
        line(x + pad, y + pad, x + cell - pad, y + cell - pad);
        line(x + cell - pad, y + pad, x + pad, y + cell - pad);
      } else if (mark === "O") {
        stroke(160, 200, 255);
        strokeWeight(LINE_WEIGHT + 1);
        noFill();
        ellipse(x + cell / 2, y + cell / 2, cell - 2 * pad, cell - 2 * pad);
      }
    }
  }
}

function netViewBoardAndPiece() {
  if (mgr.state === AppState.TRAINING && mgr.trainer.visBoard !== null) {
    return {
      board: mgr.trainer.visBoard,
      piece: mgr.trainer.visNetIsX ? "X" : "O",
    };
  } else if (mgr.state === AppState.PLAYING && mgr.humanBoard !== null) {
    return {
      board: mgr.humanBoard,
      piece: mgr.humanTurn ? mgr.humanPiece : mgr.aiPiece,
    };
  }
  return {
    board: new Board(),
    piece: "X",
  };
}

function setup() {
  const dpr = window.devicePixelRatio || 1;
  pixelDensity(min(2, dpr));
  const cnv = createCanvas(CANVAS, CANVAS);
  cnv.parent("p5-wrap");
  layoutNetNodes();
  mgr = new GameManager();
}

function draw() {
  background(22, 22, 26);

  mgr.update();

  const boardPx = 252 * S;
  const ox = (SPLIT_X - boardPx) / 2;
  const oy = (CANVAS - boardPx) / 2 - 18 * S;
  const view = netViewBoardAndPiece();
  drawBoardRegion(view.board, ox, oy, boardPx);

  stroke(50, 52, 60);
  strokeWeight(2 * S);
  line(SPLIT_X, 0, SPLIT_X, CANVAS);

  const netViz =
    mgr.state === AppState.TRAINING &&
    mgr.trainer.population.length > mgr.trainer.trainInd
      ? mgr.trainer.population[mgr.trainer.trainInd]
      : mgr.trainer.showcaseBrain;

  if (netViz !== null) {
    const inp = view.board.toInput(view.piece);
    const fh = netViz.forward(inp);
    drawNetColumnCaptions();
    drawNetworkViz(
      netViz,
      inp,
      fh.hidden,
      fh.out,
      mgr.state === AppState.TRAINING ? mgr.trainer.visBoard : null
    );
    drawNetNodeKey();
  }

  noStroke();
  textAlign(LEFT, TOP);
  fill(200, 202, 212);
  textSize(16 * S);
  text(
    "Generation: " +
      mgr.trainer.generation +
      "    State: " +
      mgr.state +
      "    Auto: " +
      (mgr.autoTrain ? "on" : "off"),
    HUD_X,
    HUD_Y_GEN
  );
  textSize(15 * S);
  text(
    "Best fitness: " + nf(mgr.trainer.bestFitnessThisGen, 0, 2),
    HUD_X,
    HUD_Y_FIT
  );

  textSize(11 * S);
  fill(120, 122, 132);
  if (mgr.state === AppState.TRAINING) {
    fill(180, 190, 210);
    let statusText = "Training gen " + (mgr.trainer.generation + 1);
    if (mgr.batchTargetGen > mgr.trainer.generation) {
      statusText += " (Batch target: " + mgr.batchTargetGen + ")";
    }
    statusText += " — brain " + (mgr.trainer.trainInd + 1) + "/" + POP;
    text(statusText, HUD_X, HUD_Y_ROW3);
  } else if (mgr.state === AppState.PLAYING) {
    fill(255, 200, 100);
    if (mgr.gameResult) {
      textSize(16 * S);
      text(mgr.gameResult + " — Press P to play again", HUD_X, HUD_Y_ROW3);
      textSize(11 * S);
    } else {
      text(
        mgr.humanTurn ? "Your turn (" + mgr.humanPiece + ")" : "AI thinking...",
        HUD_X,
        HUD_Y_ROW3
      );
    }
  } else {
    text("System Idle", HUD_X, HUD_Y_ROW3);
  }

  fill(120, 122, 132);
  text(
    "Space/N: train 1    B: train 10    M: train 50    P: play",
    HUD_X,
    HUD_Y_ROW4
  );
  text("A: auto-train toggle    R: reset evolution", HUD_X, HUD_Y_R);
  text(
    "1/2/3/4: speed (" + TRAINING_SPEED_LABELS[mgr.trainingSpeedIndex] + ")",
    HUD_X,
    HUD_Y_SPEED
  );

  textAlign(LEFT, TOP);
  textSize(11 * S);
  fill(100, 102, 115);
  const footY = CANVAS - 36 * S;
  const footX = 12 * S;
  if (mgr.state === AppState.PLAYING) {
    text(
      "Left: click the board to place your piece. Right: current champion brain.",
      footX,
      footY,
      LEFT_PANEL_TEXT_W
    );
  } else if (mgr.state !== AppState.TRAINING) {
    text(
      "Left: empty board between runs. Right: last generation’s champion network.",
      footX,
      footY,
      LEFT_PANEL_TEXT_W
    );
  } else {
    text(
      "Left: fitness game for brain " +
        (mgr.trainer.trainInd + 1) +
        " / " +
        POP +
        ".",
      footX,
      footY,
      LEFT_PANEL_TEXT_W
    );
  }
}

function mouseClicked() {
  const boardPx = 252 * S;
  const ox = (SPLIT_X - boardPx) / 2;
  const oy = (CANVAS - boardPx) / 2 - 18 * S;
  mgr.handleMouseClick(mouseX, mouseY, ox, oy, boardPx);
}

function keyPressed() {
  if (key === "r" || key === "R") {
    mgr.reset();
    return;
  }
  if (key === "p" || key === "P") {
    mgr.startHumanGame();
    return;
  }
  if (key === "b" || key === "B") {
    mgr.startBatch(10);
    return;
  }
  if (key === "m" || key === "M") {
    mgr.startBatch(50);
    return;
  }
  if (key === "a" || key === "A") {
    mgr.toggleAuto();
    return;
  }
  if (key === " " || keyCode === 32 || key === "n" || key === "N") {
    mgr.startTraining();
    return;
  }
  if (key === "]") {
    mgr.cycleSpeed(1);
    return;
  }
  if (key === "[") {
    mgr.cycleSpeed(-1);
    return;
  }
  if (key === "1") {
    mgr.setSpeed(0);
    return;
  }
  if (key === "2") {
    mgr.setSpeed(1);
    return;
  }
  if (key === "3") {
    mgr.setSpeed(2);
    return;
  }
  if (key === "4") {
    mgr.setSpeed(3);
    return;
  }
}
