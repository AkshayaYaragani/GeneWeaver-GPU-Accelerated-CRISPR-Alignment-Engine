import type { FastaContig, HardwareTopology, ParsedFastaResult, SequenceChunk } from './types.ts';

const VALID_IUPAC_DNA = /[^ACGTN]/g;

export function cleanDnaSequence(raw: string): string {
  return raw.toUpperCase().replace(/\s+/g, '').replace(VALID_IUPAC_DNA, 'N');
}

export function computeBaseStats(sequence: string): {
  baseCounts: FastaContig['baseCounts'];
  gcPercent: number;
} {
  let A = 0;
  let C = 0;
  let G = 0;
  let T = 0;
  let N = 0;

  for (let i = 0; i < sequence.length; i++) {
    const ch = sequence.charCodeAt(i);
    if (ch === 65) A++;
    else if (ch === 67) C++;
    else if (ch === 71) G++;
    else if (ch === 84) T++;
    else N++;
  }

  const totalAtgc = A + C + G + T;
  const gcPercent = totalAtgc > 0 ? Number((((G + C) / totalAtgc) * 100).toFixed(2)) : 0;

  return {
    baseCounts: { A, C, G, T, N },
    gcPercent,
  };
}

export function parseFastaContent(rawText: string, fileName = 'uploaded_genome.fasta'): ParsedFastaResult {
  const start = typeof performance !== 'undefined' ? performance.now() : Date.now();
  const lines = rawText.split(/\r?\n/);

  const contigs: FastaContig[] = [];
  let currentHeader = '';
  let currentId = '';
  let buffer: string[] = [];

  const flushContig = () => {
    if (buffer.length === 0 && !currentHeader) return;
    const sequence = cleanDnaSequence(buffer.join(''));
    if (sequence.length > 0) {
      const { baseCounts, gcPercent } = computeBaseStats(sequence);
      contigs.push({
        id: currentId || `contig_${contigs.length + 1}`,
        header: currentHeader || `contig_${contigs.length + 1}`,
        sequence,
        lengthBp: sequence.length,
        gcPercent,
        baseCounts,
      });
    }
    buffer = [];
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line || line.startsWith(';')) continue;
    if (line.startsWith('>')) {
      flushContig();
      currentHeader = line.slice(1).trim();
      const firstToken = currentHeader.split(/\s+/)[0];
      currentId = firstToken || `contig_${contigs.length + 1}`;
    } else {
      buffer.push(line);
    }
  }
  flushContig();

  // Fallback if user pasted raw DNA sequence without '>' FASTA header
  if (contigs.length === 0 && rawText.trim().length > 0) {
    const seq = cleanDnaSequence(rawText);
    if (seq.length > 0) {
      const { baseCounts, gcPercent } = computeBaseStats(seq);
      contigs.push({
        id: 'RAW_SEQ_1',
        header: 'RAW_SEQ_1 Direct DNA Sequence Input',
        sequence: seq,
        lengthBp: seq.length,
        gcPercent,
        baseCounts,
      });
    }
  }

  const totalBp = contigs.reduce((sum, c) => sum + c.lengthBp, 0);
  const totalGc = contigs.reduce((sum, c) => sum + c.baseCounts.G + c.baseCounts.C, 0);
  const totalAtgc = contigs.reduce(
    (sum, c) => sum + c.baseCounts.A + c.baseCounts.C + c.baseCounts.G + c.baseCounts.T,
    0
  );
  const overallGcPercent = totalAtgc > 0 ? Number(((totalGc / totalAtgc) * 100).toFixed(2)) : 0;
  const end = typeof performance !== 'undefined' ? performance.now() : Date.now();

  return {
    fileName,
    contigs,
    totalBp,
    overallGcPercent,
    parseTimeMs: Number(Math.max(0.05, end - start).toFixed(2)),
  };
}

/**
 * Splits contigs into overlapping memory-safe chunks so CRISPR targets straddling
 * chunk boundaries are never lost and GPU VRAM remains strictly bounded.
 */
export function chunkContigsForCluster(
  contigs: FastaContig[],
  chunkSizeBp: number,
  overlapBp: number,
  topology: HardwareTopology
): SequenceChunk[] {
  const chunks: SequenceChunk[] = [];
  const safeChunkSize = Math.max(512, chunkSizeBp);
  const safeOverlap = Math.min(Math.max(0, overlapBp), Math.floor(safeChunkSize / 4));
  const step = safeChunkSize - safeOverlap;

  let globalChunkIdx = 0;

  for (const contig of contigs) {
    const seqLen = contig.lengthBp;
    if (seqLen === 0) continue;

    let start = 0;
    while (start < seqLen) {
      const end = Math.min(seqLen, start + safeChunkSize);
      const assignedGpu: 0 | 1 =
        topology === '2x-gpu-dask' ? ((globalChunkIdx % 2) as 0 | 1) : 0;

      chunks.push({
        chunkIndex: globalChunkIdx,
        contigId: contig.id,
        startOffsetBp: start,
        endOffsetBp: end,
        lengthBp: end - start,
        sequence: contig.sequence.slice(start, end),
        assignedGpu,
      });

      globalChunkIdx++;
      if (end === seqLen) break;
      start += step;
    }
  }

  return chunks;
}
