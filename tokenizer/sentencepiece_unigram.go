package tokenizer

import (
	"fmt"
	"log/slog"
	"strconv"
	"strings"

	"github.com/ollama/ollama/logutil"
)

// SentencePieceUnigram implements the SentencePiece Unigram tokenizer
// using Viterbi dynamic programming. This produces correct tokenization
// for XLM-RoBERTa and similar models that use Unigram (not BPE) scoring.
//
// The existing SentencePiece tokenizer uses greedy pairwise merge which
// is correct for BPE-style models but wrong for Unigram models.
type SentencePieceUnigram struct {
	maxTokenLen int
	vocab       *Vocabulary
	tokenMap    map[string]int32 // token string → ID for fast lookup
}

var _ Tokenizer = (*SentencePieceUnigram)(nil)

func (spm SentencePieceUnigram) Vocabulary() *Vocabulary {
	return spm.vocab
}

func NewSentencePieceUnigram(vocab *Vocabulary) SentencePieceUnigram {
	logutil.Trace("SentencePieceUnigram", "num tokens", len(vocab.Values))

	tokenMap := make(map[string]int32, len(vocab.Values))
	var maxTokenLen int
	for i, tok := range vocab.Values {
		if vocab.Types[i] == TOKEN_TYPE_NORMAL || vocab.Types[i] == TOKEN_TYPE_USER_DEFINED {
			tokenMap[tok] = int32(i)
			if len(tok) > maxTokenLen {
				maxTokenLen = len(tok)
			}
		}
	}

	// Also add control/special tokens to the map
	for i, tok := range vocab.Values {
		if vocab.Types[i] == TOKEN_TYPE_CONTROL || vocab.Types[i] == TOKEN_TYPE_UNKNOWN {
			tokenMap[tok] = int32(i)
		}
	}

	// Add byte fallback tokens
	for i, tok := range vocab.Values {
		if vocab.Types[i] == TOKEN_TYPE_BYTE {
			tokenMap[tok] = int32(i)
		}
	}

	return SentencePieceUnigram{
		maxTokenLen: maxTokenLen,
		vocab:       vocab,
		tokenMap:    tokenMap,
	}
}

func (spm SentencePieceUnigram) Is(id int32, special Special) bool {
	return spm.vocab.Is(id, special)
}

// viterbi finds the optimal segmentation of text into tokens using
// dynamic programming. For each position, it tries all vocab tokens
// ending there and picks the segmentation with the highest total score.
func (spm SentencePieceUnigram) viterbi(text string) []int32 {
	if text == "" {
		return nil
	}

	n := len(text)

	// best[i] = best total score for text[0..i)
	// back[i] = (token_id, start_pos) for backtracking
	type backInfo struct {
		tokenID  int32
		startPos int
	}

	best := make([]float32, n+1)
	back := make([]backInfo, n+1)
	for i := range best {
		best[i] = -1e30
		back[i] = backInfo{-1, -1}
	}
	best[0] = 0

	for i := 0; i < n; i++ {
		if best[i] <= -1e29 {
			continue // unreachable position
		}

		maxLen := spm.maxTokenLen
		if n-i < maxLen {
			maxLen = n - i
		}

		for length := 1; length <= maxLen; length++ {
			end := i + length

			// Only try lengths ending on UTF-8 character boundaries
			if end < n {
				c := text[end]
				if c&0xC0 == 0x80 {
					continue // mid-sequence byte
				}
			}

			piece := text[i:end]
			id, ok := spm.tokenMap[piece]
			if !ok {
				continue
			}

			score := float32(0)
			if int(id) < len(spm.vocab.Scores) {
				score = spm.vocab.Scores[id]
			}

			candidate := best[i] + score
			if candidate > best[end] {
				best[end] = candidate
				back[end] = backInfo{id, i}
			}
		}

		// Byte fallback: ensure we can always advance
		if i+1 <= n && best[i+1] <= -1e29 {
			b := text[i]
			byteToken := fmt.Sprintf("<0x%02X>", b)
			id, ok := spm.tokenMap[byteToken]
			if !ok {
				// Use unknown token as last resort
				for j, tok := range spm.vocab.Values {
					if spm.vocab.Types[j] == TOKEN_TYPE_UNKNOWN {
						id = int32(j)
						_ = tok
						ok = true
						break
					}
				}
			}
			if ok {
				candidate := best[i] + (-100.0) // heavy penalty
				if candidate > best[i+1] {
					best[i+1] = candidate
					back[i+1] = backInfo{id, i}
				}
			}
		}
	}

	// Backtrack
	var tokens []int32
	pos := n
	for pos > 0 {
		info := back[pos]
		if info.tokenID < 0 {
			slog.Debug("viterbi: unreachable position, skipping", "pos", pos)
			pos--
			continue
		}
		tokens = append(tokens, info.tokenID)
		pos = info.startPos
	}

	// Reverse
	for i, j := 0, len(tokens)-1; i < j; i, j = i+1, j-1 {
		tokens[i], tokens[j] = tokens[j], tokens[i]
	}

	return tokens
}

func (spm SentencePieceUnigram) Encode(s string, addSpecial bool) ([]int32, error) {
	// Handle special tokens embedded in input
	fragments := []fragment{{value: s}}
	for _, special := range spm.vocab.SpecialVocabulary() {
		id := spm.vocab.Encode(special)
		for i := 0; i < len(fragments); i++ {
			frag := fragments[i]
			if len(frag.ids) > 0 {
				continue
			}

			var middle []fragment
			switch idx := strings.Index(frag.value, special); {
			case idx < 0:
				middle = append(middle, frag)
			case idx > 0:
				middle = append(middle, fragment{value: frag.value[:idx]})
				fallthrough
			default:
				middle = append(middle, fragment{value: special, ids: []int32{id}})
				if rest := frag.value[idx+len(special):]; rest != "" {
					middle = append(middle, fragment{value: rest})
				}
			}

			fragments = append(fragments[:i], append(middle, fragments[i+1:]...)...)
		}
	}

	var ids []int32
	for _, frag := range fragments {
		if len(frag.ids) > 0 {
			ids = append(ids, frag.ids...)
			continue
		}

		// SentencePiece Unigram convention: prepend space then replace all spaces with ▁
		// This turns "Hello world" into "▁Hello▁world" matching HuggingFace behavior
		text := strings.ReplaceAll(" "+frag.value, " ", spmWhitespaceSep)

		// Check if the whole text is a single token
		if id := spm.vocab.Encode(text); id >= 0 {
			ids = append(ids, id)
			continue
		}

		// Viterbi DP tokenization
		tokens := spm.viterbi(text)
		ids = append(ids, tokens...)
	}

	if addSpecial {
		ids = spm.vocab.addSpecials(ids)
	}

	logutil.Trace("encoded (unigram)", "string", s, "ids", ids)
	return ids, nil
}

func (spm SentencePieceUnigram) Decode(ids []int32) (string, error) {
	var sb strings.Builder
	for _, id := range ids {
		data := spm.vocab.Decode(id)
		data = strings.ReplaceAll(data, spmWhitespaceSep, " ")

		if len(data) == 6 && strings.HasPrefix(data, "<0x") && strings.HasSuffix(data, ">") {
			byteVal, err := strconv.ParseUint(data[1:5], 0, 8)
			if err != nil {
				return "", fmt.Errorf("failed to parse hex byte: %v", err)
			}

			if err := sb.WriteByte(byte(byteVal)); err != nil {
				return "", err
			}
		} else {
			if _, err := sb.WriteString(data); err != nil {
				return "", err
			}
		}
	}

	return sb.String(), nil
}
