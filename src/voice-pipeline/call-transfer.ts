import twilio from "twilio";

// Lazy-initialized Twilio client
let _twilioClient: ReturnType<typeof twilio> | null = null;
// Lazy-initialized SignalWire client (Twilio-compatible)
let _signalWireClient: any | null = null;

function getTwilioClient(): ReturnType<typeof twilio> {
  if (!_twilioClient) {
    const sid = process.env.TWILIO_ACCOUNT_SID;
    const token = process.env.TWILIO_AUTH_TOKEN;
    if (!sid || !token) throw new Error("TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN required for call transfer");
    _twilioClient = twilio(sid, token);
  }
  return _twilioClient;
}

function getSignalWireClient(): any {
  if (!_signalWireClient) {
    const projectId = process.env.SW_PROJECT_ID;
    const authToken = process.env.SW_AUTH_TOKEN;
    const spaceUrl = process.env.SW_SPACE_URL;
    if (!projectId || !authToken || !spaceUrl) {
      throw new Error("SW_PROJECT_ID, SW_AUTH_TOKEN, and SW_SPACE_URL required for SignalWire call transfer");
    }
    const { RestClient } = require("@signalwire/compatibility-api");
    _signalWireClient = new RestClient(projectId, authToken, { signalwireSpaceUrl: spaceUrl });
  }
  return _signalWireClient;
}

/** Get the correct provider client based on provider string */
function getProviderClient(provider?: string): any {
  if (provider === "signalwire") return getSignalWireClient();
  return getTwilioClient();
}

/** Clear cached clients (called when settings are updated) */
export function clearTransferClients(): void {
  _twilioClient = null;
  _signalWireClient = null;
}

interface TransferOptions {
  callSid: string;
  destination: string;
  publicUrl: string;
  callerNumber: string;
  timeout?: number;
  provider?: "twilio" | "signalwire";
}

/** XML-escape a string to prevent TwiML injection */
function escapeXml(str: string): string {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&apos;");
}

/** Validate a phone number (E.164 or SIP URI) */
function isValidDestination(dest: string): boolean {
  // E.164: +1234567890 (7-15 digits)
  if (/^\+[1-9]\d{6,14}$/.test(dest)) return true;
  // SIP URI
  if (/^sip:[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+$/.test(dest)) return true;
  return false;
}

/** Validate DTMF digits (0-9, *, #, w for wait) */
function isValidDTMF(digits: string): boolean {
  return /^[0-9*#wW]+$/.test(digits);
}

/**
 * Transfer an active call using <Dial> verb.
 * Works with both Twilio and SignalWire (LaML is identical to TwiML).
 */
export async function transferCallWithDial(opts: TransferOptions): Promise<{ success: boolean; error?: string }> {
  const client = getProviderClient(opts.provider);
  const webhookPrefix = opts.provider === "signalwire" ? "signalwire" : "twilio";

  // Validate destination to prevent TwiML injection
  if (!isValidDestination(opts.destination)) {
    return { success: false, error: `Invalid destination format: ${opts.destination.slice(0, 20)}` };
  }

  const timeout = Math.min(Math.max(opts.timeout || 30, 5), 120);

  try {
    await client.calls(opts.callSid).update({
      twiml: `<Response>
        <Dial callerId="${escapeXml(opts.callerNumber)}" timeout="${timeout}" action="${escapeXml(opts.publicUrl)}/api/webhooks/${webhookPrefix}/transfer-status">
          <Number>${escapeXml(opts.destination)}</Number>
        </Dial>
      </Response>`,
    });

    return { success: true };
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    // Twilio timing race: customer hung up before redirect could execute
    if (message.includes("not in-progress") || message.includes("Cannot redirect")) {
      console.log(`[call-transfer] Transfer missed (${opts.provider || "twilio"}): customer already left — ${message}`);
      return { success: false, error: `transfer-missed: ${message}` };
    }
    console.error(`[call-transfer] Dial transfer failed (${opts.provider || "twilio"}):`, message);
    return { success: false, error: message };
  }
}

/**
 * Transfer using conference bridge approach.
 * Works with both Twilio and SignalWire.
 */
export async function transferCallWithConference(opts: TransferOptions): Promise<{ success: boolean; conferenceSid?: string; error?: string }> {
  const client = getProviderClient(opts.provider);

  if (!isValidDestination(opts.destination)) {
    return { success: false, error: `Invalid destination format` };
  }

  // Conference name is server-generated -- safe from injection
  const conferenceName = `transfer-${opts.callSid}-${Date.now()}`;

  try {
    await client.calls(opts.callSid).update({
      twiml: `<Response>
        <Dial>
          <Conference waitUrl="" startConferenceOnEnter="true" endConferenceOnExit="true">${escapeXml(conferenceName)}</Conference>
        </Dial>
      </Response>`,
    });

    const outboundLeg = await client.calls.create({
      to: opts.destination,
      from: opts.callerNumber,
      twiml: `<Response>
        <Dial>
          <Conference startConferenceOnEnter="true" endConferenceOnExit="false">${escapeXml(conferenceName)}</Conference>
        </Dial>
      </Response>`,
    });

    return { success: true, conferenceSid: outboundLeg.sid };
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    console.error(`[call-transfer] Conference transfer failed (${opts.provider || "twilio"}):`, message);
    return { success: false, error: message };
  }
}

/**
 * Send DTMF tones on an active call (for IVR navigation).
 * Works with both Twilio and SignalWire.
 */
export async function sendDTMF(callSid: string, digits: string, provider?: string): Promise<void> {
  if (!isValidDTMF(digits)) {
    throw new Error(`Invalid DTMF digits: only 0-9, *, #, w allowed`);
  }

  const client = getProviderClient(provider);
  await client.calls(callSid).update({
    twiml: `<Response><Play digits="${escapeXml(digits)}"/></Response>`,
  });
}
