import {
  Callout,
  Card,
  CardBody,
  CardHeader,
  Divider,
  Grid,
  H1,
  H2,
  H3,
  Pill,
  Row,
  Stack,
  Stat,
  Table,
  Text,
} from "cursor/canvas";

export default function P4KnowledgePackCluster() {
  return (
    <Stack gap={24}>
      <Stack gap={8}>
        <Row gap={8} align="center" wrap>
          <H1>Knowledge packs + P4 cluster</H1>
          <Pill active>Ethernet, not USB-C pairs</Pill>
        </Row>
        <Text tone="secondary">
          The model talks. The pack is the knowledge. Same pack file on a
          phone, a laptop, a desktop GPU, one P4, or a school Ethernet cluster.
          No Disney-style licensed worlds in the examples.
        </Text>
      </Stack>

      <Callout tone="info" title="What changed from the 12-board USB-C drawing">
        Drop USB-C host/device pairs. On Waveshare P4-ETH, Type-C is UART for
        flash. Native USB HS is a 4-pin OTG header, not a LAN. The cluster is a
        gigabit switch plus 100 Mbps P4 ports. The physical pack on a P4 is
        microSD; USB/files is how phones and PCs carry the same pack.
      </Callout>

      <Grid columns={4} gap={12}>
        <Stat value="Pack" label="Indexed facts on SD / file" />
        <Stat value="Store" label="Upload, build, sell, download" />
        <Stat value="1 P4" label="One pack, offline" />
        <Stat value="6+ P4s" label="Many packs on Ethernet" />
      </Grid>

      <H2>Store: build once, run anywhere</H2>
      <Table
        headers={["Step", "Who", "What happens"]}
        rows={[
          [
            "1. Upload",
            "Teacher, parent, shop, OEM",
            "Curriculum PDFs, manuals, field notes, tables — their data, their rights",
          ],
          [
            "2. Build pack",
            "Knowledge store",
            "Chunk, embed/index for the PFor tokenizer, fit a size budget, write a .kpack",
          ],
          [
            "3. Get it",
            "Buyer or self",
            "Download the file, or list it in the pack store",
          ],
          [
            "4. Load",
            "Any runtime",
            "microSD in a P4, USB/file on phone or laptop, folder on a desktop GPU, or a slot on the cluster switch",
          ],
        ]}
      />

      <H2>Same pack, four places</H2>
      <Grid columns={2} gap={16}>
        <Card>
          <CardHeader trailing={<Pill size="sm">Offline</Pill>}>
            One device, one pack
          </CardHeader>
          <CardBody>
            <Stack gap={8}>
              <Text>
                Kid: Grade 6 math pack on a handheld P4, microSD in the slot,
                no radio required.
              </Text>
              <Text>
                Hike: wilderness forage pack — edible vs toxic plants for this
                region, no cell coverage.
              </Text>
              <Text>
                Boat: freshwater fishing pack — species, seasons, size limits.
              </Text>
              <Text>
                Car / furnace / radio: OEM pack is the manual and error codes
                when the cloud is gone.
              </Text>
            </Stack>
          </CardBody>
        </Card>
        <Card>
          <CardHeader trailing={<Pill size="sm" tone="success">LAN</Pill>}>
            Cluster, many packs
          </CardHeader>
          <CardBody>
            <Stack gap={8}>
              <Text>
                School closet: 6–12 P4-ETH on a cheap switch. Math, geography,
                lab safety, civics packs each live on their own card.
              </Text>
              <Text>
                Question hits a router P4. It fans out to 1–3 pack boards over
                Ethernet. Those boards return short passages. One or two
                generator P4s (the small PFor, maybe 24 layers split) write the
                answer from those passages only.
              </Text>
              <Text>
                Want more packs at once: add boards (or a USB/SD hub that looks
                like more cards), not USB-C mesh between chips.
              </Text>
            </Stack>
          </CardBody>
        </Card>
      </Grid>

      <H2>Pack catalog (examples, not licensed worlds)</H2>
      <Table
        headers={["Family", "Pack", "Typical load"]}
        rows={[
          ["School", "Grade 6 math (fractions, area)", "Kid handheld or class cluster"],
          ["School", "World geography (capitals, rivers, climate)", "Class cluster + take-home SD"],
          ["School", "Lab safety (chemicals, PPE)", "Science room cluster"],
          ["Field", "Wilderness forage (this biome)", "Single P4 in a pack"],
          ["Field", "Freshwater fishing regs", "Single P4 / phone file"],
          ["Field", "Wilderness first aid", "Single P4"],
          ["Machine", "Appliance error codes + install", "Fridge/HVAC offline P4"],
          ["Machine", "Vehicle owner + service intervals", "Dash or glovebox P4"],
          ["Desk", "Same .kpack on phone, laptop, desktop GPU", "LM Studio / local app, not a P4"],
        ]}
      />

      <H2>Cluster wiring (6+ P4-ETH)</H2>
      <Grid columns="1.2fr 1fr" gap={16}>
        <Card>
          <CardHeader>Roles, not USB pairs</CardHeader>
          <CardBody>
            <Stack gap={10}>
              <Text weight="semibold">Generator (1–2 boards)</Text>
              <Text tone="secondary">
                Small PFor weights in PSRAM. Talks. Must not invent a fact
                with no pack hit. Two boards can split layers (24L) over
                Ethernet; that is optional.
              </Text>
              <Text weight="semibold">Pack nodes (the rest)</Text>
              <Text tone="secondary">
                Each has one microSD knowledge pack and a tiny index. No need
                to clone the full net. They only search and return snippets.
              </Text>
              <Text weight="semibold">Router</Text>
              <Text tone="secondary">
                Any P4 (or a PC on the LAN at first). Embeds the question,
                picks pack IDs, waits for snippets, calls the generator.
              </Text>
            </Stack>
          </CardBody>
        </Card>
        <Card>
          <CardHeader>Switch, not a hub of USB-C</CardHeader>
          <CardBody>
            <Stack gap={8}>
              <H3>School rack</H3>
              <Text>
                Gigabit switch → six RJ45 drops → six P4-ETH. Packs: math,
                geography, history, science, civics, languages. Kids still
                take one SD home in a standalone reader.
              </Text>
              <H3>Appliance / vehicle</H3>
              <Text>
                Usually one P4, one pack, no cluster. Fleet shop can Ethernet
                several manuals at once the same way as the school closet.
              </Text>
            </Stack>
          </CardBody>
        </Card>
      </Grid>

      <H2>Question path</H2>
      <Table
        headers={["#", "Where", "Action"]}
        rows={[
          ["1", "Kid / hiker / dash", "Ask in plain language"],
          ["2", "Router", "Which packs? Math vs geography vs forage"],
          ["3", "Pack node(s)", "Search SD index, return 1–3 short passages"],
          ["4", "Generator P4 (or phone/GPU)", "Answer only from those passages, or say unknown"],
          ["5", "User", "Hears/sees the answer. Offline the whole way if no WAN"],
        ]}
      />

      <Callout tone="warning" title="What the neural net is not allowed to do">
        A pack miss is “unknown”, not a guess. That is how water stays 100 C
        and capitals stay in the gazetteer. FineWeb weights do not replace a
        geography pack.
      </Callout>

      <Divider />
      <Text tone="tertiary">
        Hardware today: Waveshare ESP32-P4-ETH, 32 MB PSRAM, microSD, 100 Mbps
        Ethernet, TCP 8742. Generator is PFor-shaped; packs are files. Phone
        and desktop GPU run the same pack format beside SmolLM or a local app.
      </Text>
    </Stack>
  );
}
