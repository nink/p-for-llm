import {
  Callout,
  Card,
  CardBody,
  CardHeader,
  Divider,
  Grid,
  H1,
  H2,
  Pill,
  Row,
  Stack,
  Stat,
  Table,
  Text,
  TodoListCard,
} from "cursor/canvas";

export default function BaseModelKnowledgePacks() {
  return (
    <Stack gap={24}>
      <Stack gap={8}>
        <Row gap={8} align="center">
          <H1>Redo: base model + knowledge packs</H1>
          <Pill active>Retrieve first</Pill>
        </Row>
        <Text tone="secondary">
          Do not put facts in the net. The pack is the knowledge. The model
          only routes, copies, and talks. That is the opposite of shipping
          PFor 180.9M as a generalist and hoping ChatML plus compress is a
          gazetteer.
        </Text>
      </Stack>

      <Callout tone="warning" title="What we would not repeat">
        Train a tiny general LLM on 12B web tokens, then paste a Canada
        article into the prompt and ask it to copy. That is how you get
        “capital of BC is Paris.” Compression and KV reuse are for lesson
        chapters, not for place facts.
      </Callout>

      <Grid columns={3} gap={16}>
        <Stat value="Pack" label="Facts, names, lessons on SD" />
        <Stat value="Retrieve" label="Always before any generate" />
        <Stat value="Model" label="Copy / refuse / tutor speak" />
      </Grid>

      <Stack gap={8}>
        <H2>Split of jobs</H2>
        <Table
          headers={["Layer", "Lives where", "Allowed to do", "Forbidden"]}
          rows={[
            [
              "Knowledge pack",
              "SD folder (swappable)",
              "Exact facts, official names, lesson text, quiz keys",
              "Being ‘remembered’ by weights",
            ],
            [
              "Retriever",
              "C on P4 (index in PSRAM/SD)",
              "Name / BM25 / slot fill → 1–3 rows or a lesson chunk",
              "Guessing when the index misses",
            ],
            [
              "Base model",
              "Fixed weights on SD",
              "Intent, copy the hit, short tutor talk, say Unknown",
              "Answering a fact with no hit",
            ],
          ]}
        />
      </Stack>

      <Grid columns={2} gap={16}>
        <Card>
          <CardHeader>Runtime loop (every question)</CardHeader>
          <CardBody>
            <Stack gap={8}>
              <Text weight="semibold">1. Classify</Text>
              <Text tone="secondary">
                Fact lookup vs lesson explain vs quiz. Rules first; a tiny
                classifier later.
              </Text>
              <Text weight="semibold">2. Retrieve</Text>
              <Text tone="secondary">
                Gazetteer key or chapter chunk. Empty hit → “Unknown — not in
                this pack.” Stop. No model.
              </Text>
              <Text weight="semibold">3. Emit or speak</Text>
              <Text tone="secondary">
                Facts: print the row (template). Explanations: model sees only
                the retrieved snippet and must copy/cite it.
              </Text>
              <Text weight="semibold">4. Session</Text>
              <Text tone="secondary">
                Prefill a lesson chunk once; follow-up questions reuse KV.
                New pack or new chapter pays TTFT again.
              </Text>
            </Stack>
          </CardBody>
        </Card>
        <Card>
          <CardHeader>Pack on SD (cartridge)</CardHeader>
          <CardBody>
            <Stack gap={8}>
              <Text>
                <Text as="span" weight="semibold">
                  pack.toml
                </Text>
                {" — id, version, locale, schema"}
              </Text>
              <Text>
                <Text as="span" weight="semibold">
                  index/
                </Text>
                {" — names, aliases, lat/lon, province (CGNDB class)"}
              </Text>
              <Text>
                <Text as="span" weight="semibold">
                  lessons/
                </Text>
                {" — short chapters for tutor mode (optional)"}
              </Text>
              <Text>
                <Text as="span" weight="semibold">
                  eval.jsonl
                </Text>
                {" — held-out questions shipped with the pack"}
              </Text>
              <Text tone="secondary">
                Swap the card, not the model. Canada, Grade 8 math, and a
                service manual are three packs on one base runtime.
              </Text>
            </Stack>
          </CardBody>
        </Card>
      </Grid>

      <Stack gap={8}>
        <H2>What to train the base model on</H2>
        <Text tone="secondary">
          Not FineWeb. Supervised pairs of (question + retrieved rows) → one
          short answer or Unknown. Include hard negatives: empty retrieve,
          homonyms, “answer using only CONTEXT.”
        </Text>
        <Table
          headers={["Objective", "Example", "Success"]}
          rows={[
            [
              "Copy",
              "Q: Oka?  CTX: municipality in Quebec  → Quebec",
              "Gold substring; no Paris/USA",
            ],
            [
              "Refuse",
              "Q: capital of Atlantis?  CTX: (empty)  → Unknown",
              "Does not invent",
            ],
            [
              "Route",
              "“what’s the code for…” → lookup, not chat",
              "Right tool / index",
            ],
            [
              "Tutor speak",
              "Lesson chunk + “why stomata?” → 1–2 sentences from chunk",
              "No facts outside snippet",
            ],
          ]}
        />
        <Text tone="secondary">
          Size: a 10–30M copy model is enough if retrieve is good. Keeping
          PFor’s 180.9M only makes sense if you have a .pt checkpoint and SFT
          it on this dialect — you cannot continue-train pfor-180m.llmcraft.
        </Text>
      </Stack>

      <Divider />

      <Stack gap={8}>
        <H2>Build order</H2>
        <TodoListCard
          defaultExpanded
          todos={[
            {
              id: "v0",
              content:
                "V0 — Pack + retrieve + template answers. No neural net in the fact path. Eval-200 >95% contain-gold.",
              status: "in_progress",
            },
            {
              id: "v1",
              content:
                "V1 — Intent/slot parser so messy English still hits the row. Still no generate for facts.",
              status: "pending",
            },
            {
              id: "v2",
              content:
                "V2 — Small speaker SFT on (question, hit) → one sentence. Firmware that hard-fails if retrieve is empty.",
              status: "pending",
            },
            {
              id: "v3",
              content:
                "V3 — Lesson packs: retrieve a chunk, prefill once, reuse KV for questions. Compression only here.",
              status: "pending",
            },
          ]}
        />
      </Stack>

      <Callout tone="success" title="Canada first cartridge">
        Keep lookup.py as the fact engine. Fix accents and homonyms. Do not
        wait on a new base model to make geography true. Add a speaker only
        after the pack already answers eval without one.
      </Callout>
    </Stack>
  );
}
