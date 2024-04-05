import { PriorityQueue } from './pqueue';
import { Node } from './node';
import { cosineSimilarity, euclideanSimilarity } from './similarity';

type Metric = 'cosine' | 'euclidean';

export class HNSW {
  metric: Metric; // Metric to use
  similarityFunction: (a: number[] | Float32Array, b: number[] | Float32Array) => number;
  d: number | null = null; // Dimension of the vectors
  M: number; // Max number of neighbors
  efConstruction: number; // Max number of nodes to visit during construction
  levelMax: number; // Max level of the graph
  entryPointId: string; // Id of the entry point
  nodes: Map<string, Node>; // Map of nodes
  probs: number[]; // Probabilities for the levels

  constructor(M = 16, efConstruction = 200, d: number | null = null, metric = 'cosine') {
    this.metric = metric as Metric;
    this.d = d;
    this.M = M;
    this.efConstruction = efConstruction;
    this.entryPointId = '';
    this.nodes = new Map<string, Node>();
    this.probs = this.set_probs(M, 1 / Math.log(M));
    this.levelMax = this.probs.length - 1;
    this.similarityFunction = this.getMetric(metric as Metric);
  }

  private getMetric(metric: Metric): (a: number[] | Float32Array, b: number[] | Float32Array) => number {
    if (metric === 'cosine') {
      return cosineSimilarity;
    } else if (metric === 'euclidean') {
      return euclideanSimilarity;
    } else {
      throw new Error('Invalid metric');
    }
  }

  private set_probs(M: number, levelMult: number): number[] {
    let level = 0;
    const probs = [];
    while (true) {
      const prob = Math.exp(-level / levelMult) * (1 - Math.exp(-1 / levelMult));
      if (prob < 1e-9) break;
      probs.push(prob);
      level++;
    }
    return probs;
  }

  private selectLevel(): number {
    let r = Math.random();
    this.probs.forEach((p, i) => {
      if (r < p) {
        return i;
      }
      r -= p;
    });
    return this.probs.length - 1;
  }

  private async addNodeToGraph(node: Node) {
    if (this.entryPointId === '') {
      this.entryPointId = node.id;
      return;
    }

    let currentNode = this.nodes.get(this.entryPointId)!;
    let closestNode = currentNode;

    for (let level = this.levelMax; level >= 0; level--) {
      while (true) {
        let nextNode = null;
        let maxSimilarity = -Infinity;

        for (const neighborId of currentNode.neighbors[level]) {
          if (neighborId === '') break;

          const neighborNode = this.nodes.get(neighborId)!;
          const similarity = this.similarityFunction(node.vector, neighborNode.vector);
          if (similarity > maxSimilarity) {
            maxSimilarity = similarity;
            nextNode = neighborNode;
          }
        }

        if (nextNode && maxSimilarity > this.similarityFunction(node.vector, closestNode.vector)) {
          currentNode = nextNode;
          closestNode = currentNode;
        } else {
          break;
        }
      }
    }

    const closestLevel = Math.min(node.level, closestNode.level);
    for (let level = 0; level <= closestLevel; level++) {
      // Add new neighbor to closestNode's neighbors
      closestNode.neighbors[level] = closestNode.neighbors[level].filter((id) => id !== '');
      closestNode.neighbors[level].push(node.id);
      // If the number of neighbors exceeds M, remove the farthest one
      if (closestNode.neighbors[level].length > this.M) {
        closestNode.neighbors[level].pop();
      }

      // Add new neighbor to node's neighbors
      node.neighbors[level] = node.neighbors[level].filter((id) => id !== '');
      node.neighbors[level].push(closestNode.id);
      // If the number of neighbors exceeds M, remove the farthest one
      if (node.neighbors[level].length > this.M) {
        node.neighbors[level].pop();
      }
    }
  }

  async addPoint(id: string, vector: Float32Array | number[]) {
    if (id === '') {
      throw new Error('Id cannot be empty');
    }
    if (this.d !== null && vector.length !== this.d) {
      throw new Error('All vectors must be of the same dimension');
    }
    this.d = vector.length;

    this.nodes.set(id, new Node(id, vector, this.selectLevel(), this.M));
    const node = this.nodes.get(id)!;
    this.levelMax = Math.max(this.levelMax, node.level);

    await this.addNodeToGraph(node);
  }

  searchKNN(query: Float32Array | number[], k: number): { id: string; score: number }[] {
    const result: { id: string; score: number }[] = [];
    const visited: Set<string> = new Set<string>();

    const candidates = new PriorityQueue<string>((a, b) => {
      const aNode = this.nodes.get(a)!;
      const bNode = this.nodes.get(b)!;
      return this.similarityFunction(query, bNode.vector) - this.similarityFunction(query, aNode.vector);
    });

    candidates.push(this.entryPointId);
    let level = this.levelMax;

    while (!candidates.isEmpty() && result.length < k) {
      const currentId = candidates.pop()!;
      if (visited.has(currentId)) continue;

      visited.add(currentId);

      const currentNode = this.nodes.get(currentId)!;
      const similarity = this.similarityFunction(currentNode.vector, query);

      if (similarity > 0) {
        result.push({ id: currentId, score: similarity });
      }

      if (currentNode.level === 0) {
        continue;
      }

      level = Math.min(level, currentNode.level - 1);

      for (let i = level; i >= 0; i--) {
        const neighbors = currentNode.neighbors[i];
        for (const neighborId of neighbors) {
          if (!visited.has(neighborId)) {
            candidates.push(neighborId);
          }
        }
      }
    }

    return result.slice(0, k);
  }

  async buildIndex(data: { id: string; vector: Float32Array | number[] }[]) {
    // Clear existing index
    this.nodes.clear();
    this.levelMax = 0;
    this.entryPointId = '';

    // Add points to the index
    for (const item of data) {
      await this.addPoint(item.id, item.vector);
    }
  }

  toJSON() {
    const entries = Array.from(this.nodes.entries());
    return {
      M: this.M,
      efConstruction: this.efConstruction,
      levelMax: this.levelMax,
      entryPointId: this.entryPointId,
      nodes: entries.map(([id, node]) => {
        return [
          id,
          {
            id: node.id,
            level: node.level,
            vector: Array.from(node.vector),
            neighbors: node.neighbors.map((level) => Array.from(level)),
          },
        ];
      }),
    };
  }

  static fromJSON(json: any): HNSW {
    const hnsw = new HNSW(json.M, json.efConstruction);
    hnsw.levelMax = json.levelMax;
    hnsw.entryPointId = json.entryPointId;
    hnsw.nodes = new Map(
      json.nodes.map(([id, node]: [number, any]) => {
        return [
          id,
          {
            ...node,
            vector: new Float32Array(node.vector),
          },
        ];
      }),
    );
    return hnsw;
  }
}
