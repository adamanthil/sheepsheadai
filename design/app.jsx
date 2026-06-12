/* ============================================================
   App composition — design canvas with 12 artboards + tweaks
   ============================================================ */

const DEFAULTS = /*EDITMODE-BEGIN*/{
  "cardStyle": "classic",
  "density": "comfortable",
  "showAccents": true
}/*EDITMODE-END*/;

function App() {
  const [t, setTweak] = useTweaks(DEFAULTS);

  // Density → CSS variable
  React.useEffect(() => {
    const map = { compact: 0.78, comfortable: 1, spacious: 1.18 };
    document.documentElement.style.setProperty('--density', map[t.density] || 1);
  }, [t.density]);

  // Navigate inside the canvas: we use a global event so artboards can request
  // their cousin artboard be focused. The canvas doesn't expose an imperative
  // focus API, so we degrade to a no-op (visual button press only).
  const navigate = React.useCallback((key) => {
    // no-op in static — users can open any artboard in focus mode (click label)
  }, []);

  const cardStyle = t.cardStyle;

  return (
    <>
      <DesignCanvas backgroundCss="#e8e0c9">
        <DCSection id="home" title="Home" subtitle="Hero with wordmark, 'how to play' inline, lobby. Heirloom — cream paper, paprika & sage.">
          <DCArtboard id="home-d" label="Desktop · 1440 × 900" width={1440} height={900}>
            <HomePage viewport="desktop" navigate={navigate} theme="heirloom" />
          </DCArtboard>
          <DCArtboard id="home-m" label="Mobile · 390 × 780" width={390} height={780}>
            <HomePage viewport="mobile" navigate={navigate} theme="heirloom" />
          </DCArtboard>
        </DCSection>

        <DCSection id="waiting" title="Waiting Room" subtitle="Seats as place cards on desktop, single-row list on mobile. Editorial captions for rule modes.">
          <DCArtboard id="wait-d" label="Desktop · 1440 × 900" width={1440} height={900}>
            <WaitingPage viewport="desktop" navigate={navigate} theme="heirloom" />
          </DCArtboard>
          <DCArtboard id="wait-m" label="Mobile · 390 × 780" width={390} height={780}>
            <WaitingPage viewport="mobile" navigate={navigate} theme="heirloom" />
          </DCArtboard>
        </DCSection>

        <DCSection id="table" title="Game Table" subtitle="Mid-trick: Kyle led the Jack of Spades; you must follow trump. Right rail = scoreboard / hand history / chat.">
          <DCArtboard id="tbl-d" label="Desktop · 1440 × 900" width={1440} height={900}>
            <TablePage viewport="desktop" navigate={navigate} theme="heirloom" cardStyle={cardStyle} />
          </DCArtboard>
          <DCArtboard id="tbl-m" label="Mobile · 390 × 780" width={390} height={780}>
            <TablePage viewport="mobile" navigate={navigate} theme="heirloom" cardStyle={cardStyle} />
          </DCArtboard>
        </DCSection>

        <DCSection id="alt-broadsheet" title="Alt · Broadsheet (for reference)" subtitle="The second direction we explored — newsprint white, forest accent, modernist cards. Kept here as a comparison.">
          <DCArtboard id="bs-home-d" label="Home · Desktop" width={1440} height={900}>
            <HomePage viewport="desktop" navigate={navigate} theme="broadsheet" />
          </DCArtboard>
          <DCArtboard id="bs-tbl-d" label="Table · Desktop" width={1440} height={900}>
            <TablePage viewport="desktop" navigate={navigate} theme="broadsheet" cardStyle={cardStyle} />
          </DCArtboard>
        </DCSection>
      </DesignCanvas>

      <TweaksPanel title="Tweaks">
        <TweakSection label="Cards">
          <TweakRadio
            label="Style"
            value={t.cardStyle}
            onChange={(v) => setTweak('cardStyle', v)}
            options={[
              { value: 'classic', label: 'Classic' },
              { value: 'modern',  label: 'Modern' },
            ]}
          />
        </TweakSection>
        <TweakSection label="Layout">
          <TweakSelect
            label="Density"
            value={t.density}
            onChange={(v) => setTweak('density', v)}
            options={[
              { value: 'compact',     label: 'Compact' },
              { value: 'comfortable', label: 'Comfortable (default)' },
              { value: 'spacious',    label: 'Spacious' },
            ]}
          />
        </TweakSection>
        <TweakSection label="Tips">
          <div style={{ fontFamily: 'var(--font-display)', fontStyle: 'italic', fontSize: 13, color: '#555', lineHeight: 1.5 }}>
            Click any artboard's label to open it fullscreen. Drag to reorder. The Tweaks above update both directions live.
          </div>
        </TweakSection>
      </TweaksPanel>
    </>
  );
}

ReactDOM.createRoot(document.getElementById('root')).render(<App />);
