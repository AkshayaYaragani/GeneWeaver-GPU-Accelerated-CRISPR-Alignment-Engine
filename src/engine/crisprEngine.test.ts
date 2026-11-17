import test from 'node:test';
import assert from 'node:assert/strict';
import { chunkContigsForCluster, parseFastaContent } from './fastaParser.ts';
import {
  estimateVramBudget,
  evaluateBiologicalScore,
  reverseComplement,
  runFullCrisprScan,
  runLiveComparativeBenchmark,
  scanSequenceSequentialCpu,
  validateSgRna,
} from './crisprEngine.ts';
import { buildPresetFasta } from './presets.ts';

test('validateSgRna accepts valid 20-nt spacer and rejects invalid characters or length', () => {
  const ok = validateSgRna('gtcagatcctagcgtcgagc');
  assert.equal(ok.valid, true);
  assert.equal(ok.normalized, 'GTCAGATCCTAGCGTCGAGC');

  const badBase = validateSgRna('GTCAGATCCTAGCGTCGAGX');
  assert.equal(badBase.valid, false);

  const tooShort = validateSgRna('GTCAGATCCT');
  assert.equal(tooShort.valid, false);
});

test('parseFastaContent cleans IUPAC whitespace and computes GC percent accurately', () => {
  const fasta = `>chr17_test Homo sapiens
  ATGC atgc
  GGCC ggcc
  >chr2_test
  AAAA TTTT`;
  const parsed = parseFastaContent(fasta, 'test.fasta');
  assert.equal(parsed.contigs.length, 2);
  assert.equal(parsed.contigs[0].id, 'chr17_test');
  assert.equal(parsed.contigs[0].sequence, 'ATGCATGCGGCCGGCC');
  assert.equal(parsed.contigs[0].lengthBp, 16);
  assert.equal(parsed.contigs[0].gcPercent, 75);
  assert.equal(parsed.totalBp, 24);
});

test('chunkContigsForCluster preserves overlap and evenly balances across 2 GPUs', () => {
  const parsed = parseFastaContent(`>chr1\n${'ACGT'.repeat(500)}`, 'chunk_test.fasta');
  const chunks = chunkContigsForCluster(parsed.contigs, 600, 22, '2x-gpu-dask');
  assert.ok(chunks.length >= 4);
  assert.equal(chunks[0].assignedGpu, 0);
  assert.equal(chunks[1].assignedGpu, 1);
  assert.equal(chunks[2].assignedGpu, 0);
  // Second chunk starts 22 bp before end of first chunk
  assert.equal(chunks[1].startOffsetBp, 600 - 22);
});

test('biological scoring penalizes PAM-proximal seed mismatches more heavily than distal mismatches', () => {
  const onTarget = evaluateBiologicalScore([], 20, 'NGG', 'AGG');
  assert.equal(onTarget.severity, 'On-Target');
  assert.equal(onTarget.cfdRiskScore, 1.0);

  const distalMismatch = evaluateBiologicalScore([1], 20, 'NGG', 'AGG');
  const seedMismatch = evaluateBiologicalScore([18], 20, 'NGG', 'AGG');

  assert.equal(distalMismatch.distalMismatchCount, 1);
  assert.equal(distalMismatch.seedMismatchCount, 0);
  assert.equal(seedMismatch.seedMismatchCount, 1);
  assert.ok(distalMismatch.cfdRiskScore > seedMismatch.cfdRiskScore);
});

test('scanSequenceSequentialCpu and runFullCrisprScan detect identical forward & reverse-complement off-targets', () => {
  const sgRna = 'GTCAGATCCTAGCGTCGAGC';
  const fwdSite = `${sgRna}AGG`;
  const revSite = reverseComplement(`ATCAGATCCTAGCGTCGAGC` + 'TGG'); // 1 mismatch at pos 0 on - strand
  const sequence = `${'A'.repeat(100)}${fwdSite}${'C'.repeat(150)}${revSite}${'T'.repeat(100)}`;

  const cpu = scanSequenceSequentialCpu(sequence, 'CHR_TEST', sgRna, 'NGG', 2);
  assert.equal(cpu.hits.length, 2);
  assert.ok(cpu.hits.some((h) => h.strand === '+' && h.mismatchCount === 0));
  assert.ok(cpu.hits.some((h) => h.strand === '-' && h.mismatchCount === 1));

  const parsed = parseFastaContent(`>CHR_TEST\n${sequence}`, 'inline.fasta');
  const full = runFullCrisprScan(parsed, {
    sgRna,
    pamPattern: 'NGG',
    maxMismatches: 2,
    chunkSizeBp: 512,
    topology: '2x-gpu-dask',
    sharedMemoryEnabled: true,
  });

  assert.equal(full.hits.length, 2);
  assert.equal(full.vramBudget.oomRisk, 'Safe');
});

test('preset FASTA scan and live benchmark return deterministic verified results', () => {
  const { fileName, content } = buildPresetFasta('chr17-tp53');
  const parsed = parseFastaContent(content, fileName);
  const summary = runFullCrisprScan(parsed, {
    sgRna: 'GTCAGATCCTAGCGTCGAGC',
    pamPattern: 'NGG',
    maxMismatches: 3,
    chunkSizeBp: 65536,
    topology: '2x-gpu-dask',
    sharedMemoryEnabled: true,
  });

  assert.equal(summary.totalBpScanned, 120_000);
  assert.ok(summary.hits.length >= 6);
  assert.ok(summary.gpu0Chunks > 0 && summary.gpu1Chunks > 0);

  const bench = runLiveComparativeBenchmark(
    parsed.contigs[0].sequence,
    'GTCAGATCCTAGCGTCGAGC',
    'NGG',
    3
  );
  assert.equal(bench.hitsVerifiedEqual, true);
  assert.ok(bench.speedupSharedVsCpu > 100);

  const vram = estimateVramBudget(1_048_576, 3_200_000_000, '2x-gpu-dask', true);
  assert.equal(vram.oomRisk, 'Safe');
  assert.ok(vram.unChunkedFullGenomeGb > 15);
});
