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
/** Plies simulated per draw() frame at each speed tier (approx. at 60fps: fast ~1s/gen, medium ~12s, slow ~minutes). */
const TRAINING_SPEED_PLIES = [1, 12, 160];
const TRAINING_SPEED_LABELS = ["slow", "medium", "fast"];
const AUTO_GEN_INTERVAL_MS = 1200;

const LINE_WEIGHT = 2 * S;
const PAD_FR = 0.18;

// --- Board / game (no training globals) ------------------------------------

function emptyBoard() {
  return [
    [null, null, null],
    [null, null, null],
    [null, null, null],
  ];
}

function boardFullBoard(board) {
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      if (board[r][c] === null) return false;
    }
  }
  return true;
}

function checkWinOnBoard(board, player) {
  for (let r = 0; r < 3; r++) {
    if (
      board[r][0] === player &&
      board[r][1] === player &&
      board[r][2] === player
    ) {
      return true;
    }
  }
  for (let c = 0; c < 3; c++) {
    if (
      board[0][c] === player &&
      board[1][c] === player &&
      board[2][c] === player
    ) {
      return true;
    }
  }
  if (
    board[0][0] === player &&
    board[1][1] === player &&
    board[2][2] === player
  ) {
    return true;
  }
  if (
    board[0][2] === player &&
    board[1][1] === player &&
    board[2][0] === player
  ) {
    return true;
  }
  return false;
}

function boardToInput(board, myPiece) {
  const input = [];
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      const cell = board[r][c];
      if (cell === null) input.push(0);
      else if (cell === myPiece) input.push(1);
      else input.push(-1);
    }
  }
  return input;
}

function randomLegalIndex(board) {
  const empties = [];
  for (let i = 0; i < 9; i++) {
    const r = Math.floor(i / 3);
    const c = i % 3;
    if (board[r][c] === null) empties.push(i);
  }
  return empties[Math.floor(random(empties.length))];
}

function pickNetMove(net, board, myPiece) {
  const input = boardToInput(board, myPiece);
  const { out } = net.forward(input);
  let best = -1;
  let bestScore = -Infinity;
  for (let i = 0; i < 9; i++) {
    const r = Math.floor(i / 3);
    const c = i % 3;
    if (board[r][c] !== null) continue;
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
}

// --- Genetic algorithm state ------------------------------------------------

let population = [];
let generation = 0;
let bestFitnessThisGen = 0;
let showcaseBrain = null;

let autoTrain = false;
let lastAutoGenMs = 0;

let trainingSpeedIndex = 0;

function initPopulation() {
  population = [];
  for (let i = 0; i < POP; i++) {
    population.push(new NeuralNet());
  }
  if (population.length > 0) {
    showcaseBrain = population[0].copy();
  }
}

function completeBreeding(fitnesses) {
  const idx = [];
  for (let i = 0; i < POP; i++) {
    idx.push(i);
  }
  idx.sort(function (a, b) {
    return fitnesses[b] - fitnesses[a];
  });

  const sorted = [];
  for (let k = 0; k < idx.length; k++) {
    sorted.push(population[idx[k]]);
  }

  bestFitnessThisGen = fitnesses[idx[0]];
  showcaseBrain = sorted[0].copy();

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
  population = next;
  generation++;
}

// --- Visual training (real eval games, stepped each frame) -------------------

let trainingActive = false;
let fitnessesWorking = [];
let trainInd = 0;
let trainGame = 0;
let visWins = 0;
let visTies = 0;
let visBoard = null;
let visTurn = "X";
let visNetIsX = true;

function resetVisualGame() {
  visBoard = emptyBoard();
  visNetIsX = random() < 0.5;
  visTurn = "X";
}

function startGenerationTraining() {
  if (trainingActive) return;
  trainingActive = true;
  trainInd = 0;
  trainGame = 0;
  visWins = 0;
  visTies = 0;
  fitnessesWorking = [];
  for (let i = 0; i < POP; i++) {
    fitnessesWorking.push(0);
  }
  resetVisualGame();
}

function finishCurrentTrainingGame() {
  trainGame++;
  if (trainGame >= GAMES_PER_BRAIN) {
    fitnessesWorking[trainInd] =
      visWins * WIN_SCORE + visTies * TIE_SCORE;
    trainInd++;
    trainGame = 0;
    visWins = 0;
    visTies = 0;
    if (trainInd >= POP) {
      completeBreeding(fitnessesWorking);
      trainingActive = false;
      visBoard = emptyBoard();
      visTurn = "X";
      lastAutoGenMs = millis();
      return;
    }
  }
  resetVisualGame();
}

function advanceVisualTrainingPly() {
  if (!trainingActive || visBoard === null) return;

  const net = population[trainInd];
  const isNet =
    (visTurn === "X" && visNetIsX) || (visTurn === "O" && !visNetIsX);
  if (isNet) {
    let idx = pickNetMove(net, visBoard, visTurn);
    if (idx < 0) {
      idx = randomLegalIndex(visBoard);
    }
    const r = Math.floor(idx / 3);
    const c = idx % 3;
    visBoard[r][c] = visTurn;
  } else {
    const idx = randomLegalIndex(visBoard);
    const r = Math.floor(idx / 3);
    const c = idx % 3;
    visBoard[r][c] = visTurn;
  }
  if (checkWinOnBoard(visBoard, visTurn)) {
    if (isNet) {
      visWins++;
    }
    finishCurrentTrainingGame();
    return;
  }
  if (boardFullBoard(visBoard)) {
    visTies++;
    finishCurrentTrainingGame();
    return;
  }
  visTurn = visTurn === "X" ? "O" : "X";
}

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
      if (boardForLegal[r][c] === null) {
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
      const mark = board[r][c];
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
  if (trainingActive && visBoard !== null) {
    return {
      board: visBoard,
      piece: visNetIsX ? "X" : "O",
    };
  }
  return {
    board: emptyBoard(),
    piece: "X",
  };
}

function setup() {
  const dpr = window.devicePixelRatio || 1;
  pixelDensity(min(2, dpr));
  const cnv = createCanvas(CANVAS, CANVAS);
  cnv.parent("p5-wrap");
  layoutNetNodes();
  initPopulation();
  visBoard = emptyBoard();
  visTurn = "X";
  lastAutoGenMs = millis();
}

function draw() {
  background(22, 22, 26);

  if (
    autoTrain &&
    !trainingActive &&
    millis() - lastAutoGenMs >= AUTO_GEN_INTERVAL_MS
  ) {
    startGenerationTraining();
  }

  if (trainingActive) {
    const plies = TRAINING_SPEED_PLIES[trainingSpeedIndex];
    for (let p = 0; p < plies; p++) {
      advanceVisualTrainingPly();
      if (!trainingActive) {
        break;
      }
    }
  }

  const boardPx = 252 * S;
  const ox = (SPLIT_X - boardPx) / 2;
  const oy = (CANVAS - boardPx) / 2 - 18 * S;
  const view = netViewBoardAndPiece();
  drawBoardRegion(view.board, ox, oy, boardPx);

  stroke(50, 52, 60);
  strokeWeight(2 * S);
  line(SPLIT_X, 0, SPLIT_X, CANVAS);

  const netViz =
    trainingActive && population.length > trainInd
      ? population[trainInd]
      : showcaseBrain;

  if (netViz !== null) {
    const inp = boardToInput(view.board, view.piece);
    const fh = netViz.forward(inp);
    drawNetColumnCaptions();
    drawNetworkViz(
      netViz,
      inp,
      fh.hidden,
      fh.out,
      trainingActive ? visBoard : null
    );
    drawNetNodeKey();
  }

  noStroke();
  textAlign(LEFT, TOP);
  fill(200, 202, 212);
  textSize(16 * S);
  text(
    "Generation: " +
      generation +
      "    Auto: " +
      (autoTrain ? "on" : "off") +
      "    Train: " +
      TRAINING_SPEED_LABELS[trainingSpeedIndex],
    HUD_X,
    HUD_Y_GEN
  );
  textSize(15 * S);
  text("Best fitness: " + nf(bestFitnessThisGen, 0, 2), HUD_X, HUD_Y_FIT);

  textSize(11 * S);
  fill(120, 122, 132);
  if (trainingActive) {
    fill(180, 190, 210);
    text(
      "Training for gen " +
        (generation + 1) +
        " — brain " +
        (trainInd + 1) +
        "/" +
        POP +
        " — eval game " +
        (trainGame + 1) +
        "/" +
        GAMES_PER_BRAIN,
      HUD_X,
      HUD_Y_ROW3
    );
    fill(120, 122, 132);
    text(
      "Watch: same brain plays vs random until fitness is tallied.",
      HUD_X,
      HUD_Y_ROW4
    );
  } else {
    text(
      "Idle — Press Space / N to train & evolve one generation",
      HUD_X,
      HUD_Y_ROW3
    );
    text("A: auto train (toggle)", HUD_X, HUD_Y_ROW4);
  }
  text("R: reset evolution", HUD_X, HUD_Y_R);
  text(
    "] / [ : training speed   1/2/3 : slow/med/fast",
    HUD_X,
    HUD_Y_SPEED
  );

  textAlign(LEFT, TOP);
  textSize(11 * S);
  fill(100, 102, 115);
  const footY = CANVAS - 36 * S;
  const footX = 12 * S;
  if (!trainingActive) {
    text(
      "Left: empty board between runs. Right: last generation’s champion network.",
      footX,
      footY,
      LEFT_PANEL_TEXT_W
    );
  } else {
    text(
      "Left: fitness game for brain " + (trainInd + 1) + " / " + POP + ".",
      footX,
      footY,
      LEFT_PANEL_TEXT_W
    );
  }
}

function keyPressed() {
  if (key === "r" || key === "R") {
    generation = 0;
    bestFitnessThisGen = 0;
    trainingActive = false;
    initPopulation();
    visBoard = emptyBoard();
    lastAutoGenMs = millis();
    return;
  }
  if (key === "a" || key === "A") {
    autoTrain = !autoTrain;
    lastAutoGenMs = millis();
    return;
  }
  if (key === " " || keyCode === 32 || key === "n" || key === "N") {
    if (!trainingActive) {
      startGenerationTraining();
    }
    return;
  }
  if (key === "]") {
    trainingSpeedIndex =
      (trainingSpeedIndex + 1) % TRAINING_SPEED_LABELS.length;
    return;
  }
  if (key === "[") {
    trainingSpeedIndex =
      (trainingSpeedIndex + TRAINING_SPEED_LABELS.length - 1) %
      TRAINING_SPEED_LABELS.length;
    return;
  }
  if (key === "1") {
    trainingSpeedIndex = 0;
    return;
  }
  if (key === "2") {
    trainingSpeedIndex = 1;
    return;
  }
  if (key === "3") {
    trainingSpeedIndex = 2;
    return;
  }
}
