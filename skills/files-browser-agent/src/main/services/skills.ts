
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { unzipSync } from 'fflate';
import type { Dirent } from 'node:fs';


export const SKILLS_ROOT =
  process.env.FILE_BROWSER_SKILLS_DIR?.trim() ||
  path.join(os.homedir(), '.file-browser-agent', 'skills');

type SkillFrontmatter = {
  name?: string;
  description?: string;
};

export type Skill = {
    name: string;
    description: string;
    body: string;
    location: string;
}

let SKILLS: Skill[] = [];

function parseFrontmatter(markdown: string): SkillFrontmatter {
  const frontmatterMatch = markdown.match(/^---\s*\r?\n([\s\S]*?)\r?\n---/);

  if (!frontmatterMatch) {
    return {};
  }

  const frontmatter = frontmatterMatch[1];
  const result: SkillFrontmatter = {};

  for (const line of frontmatter.split(/\r?\n/)) {
    const trimmed = line.trim();

    if (!trimmed || trimmed.startsWith('#')) {
      continue;
    }

    const separatorIndex = trimmed.indexOf(':');

    if (separatorIndex === -1) {
      continue;
    }

    const key = trimmed.slice(0, separatorIndex).trim();
    const value = trimmed
      .slice(separatorIndex + 1)
      .trim()
      .replace(/^['"]|['"]$/g, '');

    if (key === 'name') {
      result.name = value;
    }

    if (key === 'description') {
      result.description = value;
    }
  }

  return result;
}

export const SKILLS_STORAGE = {
    reload: async () => {
        SKILLS = await loadSkills();
        return SKILLS;
    },
    getList: () => { return SKILLS; },
    getBulletList: () => {
        return SKILLS.map((skill) => {
        return `- name: ${skill.name}\n description: ${skill.description}\n location: ${skill.location}`;
        }).join("\n")
    },
    getByName: (name: string) => {
        return SKILLS.find(s => s.name === name)
    }
}

export async function loadSkills(): Promise<Skill[]> {
const skills: Skill[] = [];

let entries: Dirent[];
try {
    entries = await fs.readdir(SKILLS_ROOT, { withFileTypes: true })
} catch {
    return skills;
}

for (const entry of entries) {
    if (!entry.isDirectory()) {
        continue;
    }

    const skillDir = path.join(SKILLS_ROOT, entry.name);

    let skillContent: string;
    try {
        skillContent = await fs.readFile(path.join(skillDir, 'SKILL.md'), 'utf8');
    } catch {
        continue;
    }

    const match = skillContent.match(/^---\s*\r?\n([\s\S]*?)\r?\n---\s*\r?\n?([\s\S]*)$/);
    if (!match) {
        continue;
    }

    const frontmatter = match[1];
    const body = match[2];

    const name = frontmatter.match(/^name:\s*(.+)$/m)?.[1]?.trim().replace(/^['"]|['"]$/g, '')
    const description = frontmatter.match(/^description:\s*(.+)$/m)?.[1]?.trim().replace(/^['"]|['"]$/g, '')

    if (!name || !description) {
        continue;
    }

    skills.push({
        name,
        description,
        body,
        location: skillDir
    })
}
return skills;
}


function isValidSkillName(name: string): boolean {
  return /^[a-zA-Z0-9_-]+$/.test(name);
}

export type InstalledSkill = { name: string; description: string; path: string };

/**
 * Installs a skill from a .zip archive into SKILLS_ROOT.
 *
 * A skill is a folder containing a SKILL.md. The zip may either wrap the files
 * in a top-level folder (`my-skill/SKILL.md`, `my-skill/script.py`, …) or place
 * SKILL.md at its root — both are accepted. The folder name is taken from the
 * SKILL.md frontmatter `name`, falling back to the wrapping folder's name.
 *
 * Throws on: missing/invalid SKILL.md, an invalid skill name, a name that is
 * already installed, or any entry that would escape the target folder (zip-slip).
 */
export async function installSkillFromZip(zipPath: string): Promise<InstalledSkill> {
  const buf = await fs.readFile(zipPath);

  let files: Record<string, Uint8Array>;
  try {
    files = unzipSync(new Uint8Array(buf));
  } catch (e) {
    throw new Error(`Not a valid .zip file: ${e instanceof Error ? e.message : String(e)}`);
  }

  // Normalise separators and locate the shallowest SKILL.md — that file's
  // directory is the root of the skill inside the archive.
  const norm = (p: string): string => p.replace(/\\/g, '/');
  let skillMdKey: string | null = null;
  for (const key of Object.keys(files)) {
    const n = norm(key);
    if (key.endsWith('/')) continue; // directory entry
    if (n === 'SKILL.md' || n.endsWith('/SKILL.md')) {
      if (skillMdKey === null || n.split('/').length < norm(skillMdKey).split('/').length) {
        skillMdKey = key;
      }
    }
  }

  if (!skillMdKey) {
    throw new Error('The zip does not contain a SKILL.md file.');
  }

  const skillMd = norm(skillMdKey);
  const prefix = skillMd.slice(0, skillMd.length - 'SKILL.md'.length); // '' or 'folder/'

  const markdown = new TextDecoder('utf-8').decode(files[skillMdKey]);
  const fallbackName = prefix.replace(/\/$/, '').split('/').pop() || '';
  const name = (parseFrontmatter(markdown).name?.trim() || fallbackName).trim();

  if (!isValidSkillName(name)) {
    throw new Error(
      `Invalid skill name "${name}". Names may only contain letters, numbers, hyphens and underscores.`
    );
  }

  const targetDir = path.join(SKILLS_ROOT, name);
  try {
    await fs.access(targetDir);
    throw new Error(`A skill named "${name}" is already installed. Remove it first to reinstall.`);
  } catch (e) {
    // ENOENT means the folder is free — anything else is a real error.
    if ((e as NodeJS.ErrnoException).code !== 'ENOENT') throw e;
  }

  await fs.mkdir(SKILLS_ROOT, { recursive: true });

  for (const [key, data] of Object.entries(files)) {
    if (key.endsWith('/')) continue; // directory entry
    const n = norm(key);
    if (prefix && !n.startsWith(prefix)) continue; // file outside the skill folder
    const rel = prefix ? n.slice(prefix.length) : n;
    if (!rel) continue;

    const dest = path.join(targetDir, rel);
    // Zip-slip guard: every written path must stay inside targetDir.
    const rootWithSep = targetDir.endsWith(path.sep) ? targetDir : targetDir + path.sep;
    if (dest !== targetDir && !dest.startsWith(rootWithSep)) {
      throw new Error(`Refusing to extract entry outside the skill folder: ${key}`);
    }

    await fs.mkdir(path.dirname(dest), { recursive: true });
    await fs.writeFile(dest, data);
  }

  const description = parseFrontmatter(markdown).description?.trim() || '';
  return { name, description, path: targetDir };
}