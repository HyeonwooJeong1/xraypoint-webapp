// Validate the release descriptor against the files that will actually be deployed.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '..');
const read = name => fs.readFileSync(path.join(root, name), 'utf8');
const release = JSON.parse(read('version.json'));
const html = read('index.html'), changelog = read('CHANGELOG.md');
const semver = /^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$/;
assert.match(release.version, semver, 'A stable MAJOR.MINOR.PATCH version is required');
assert.match(release.released, /^\d{4}-\d{2}-\d{2}$/);
assert.equal(new Date(release.released+'T00:00:00Z').toISOString().slice(0,10),release.released);
assert.equal(release.model_release, JSON.parse(read('models/meta.json')).release, 'Model release differs from the actual metadata');
assert.ok(html.includes(`<meta name="xraypoint-version" content="${release.version}">`), 'HTML version differs');
const link = html.match(/<a\b[^>]*\bid="app-version"[^>]*>([^<]*)<\/a>/);
assert.ok(link && link[1].startsWith(`v${release.version} ·`), 'Visible version differs');
assert.ok(link[0].includes(`/blob/v${release.version}/CHANGELOG.md`), 'Changelog link must point to this immutable version');
for (const script of ['lib/ort.min.js','ensemble.js','app.js']) {
  assert.ok(html.includes(`src="${script}?v=${release.version}"`), `Script cache version differs: ${script}`);
}
const firstRelease = changelog.match(/^## \[([^\]]+)\] - (\d{4}-\d{2}-\d{2})$/m);
assert.ok(firstRelease, 'Missing latest changelog entry');
assert.equal(firstRelease[1], release.version);
assert.equal(firstRelease[2], release.released);
const args = process.argv.slice(2);
assert.ok(args.length===0 || (args.length===2 && args[0]==='--previous'), 'Usage: node tools/check-release.cjs [--previous MAJOR.MINOR.PATCH]');
if (args.length) {
  assert.match(args[1], semver);
  const next = release.version.split('.').map(Number), previous = args[1].split('.').map(Number);
  const difference = next.findIndex((n,i)=>n!==previous[i]);
  assert.ok(difference>=0 && next[difference]>previous[difference], 'Version must increase from the deployed version');
}
console.log(JSON.stringify({status:'passed',version:release.version,released:release.released,model_release:release.model_release}));
