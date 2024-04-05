export class Node {
  id: string;
  level: number;
  vector: Float32Array | number[];
  neighbors: string[][];

  constructor(id: string, vector: Float32Array | number[], level: number, M: number) {
    this.id = id;
    this.vector = vector;
    this.level = level;
    this.neighbors = Array.from({ length: level + 1 }, () => new Array(M).fill(''));
  }
}
